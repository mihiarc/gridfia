"""
REST API client for FIA BIGMAP ImageServer access.
"""

import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import rasterio
from rasterio.io import MemoryFile
import numpy as np
from rich.console import Console
from rich.progress import Progress, track

from ..console import print_info, print_success, print_error, print_warning
from ..exceptions import APIConnectionError, SpeciesNotFound, DownloadError

console = Console()


class BigMapRestClient:
    """Client for accessing FIA BIGMAP ImageServer REST API with proper retry and rate limiting."""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 1.0, 
                 timeout: int = 30, rate_limit_delay: float = 0.5):
        """
        Initialize the REST client with retry and rate limiting configuration.
        
        Args:
            max_retries: Maximum number of retries for failed requests
            backoff_factor: Backoff factor for retry delays
            timeout: Request timeout in seconds
            rate_limit_delay: Delay between requests in seconds
        """
        self.base_url = "https://di-usfsdata.img.arcgis.com/arcgis/rest/services/FIA_BIGMAP_2018_Tree_Species_Aboveground_Biomass/ImageServer"
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time = 0
        
        # Configure session with retry strategy
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry
            allowed_methods=["HEAD", "GET", "OPTIONS"],  # Only retry safe methods
            raise_on_status=False  # Don't raise on HTTP errors, let us handle them
        )
        
        # Configure adapter with retry strategy
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        self.session.headers.update({
            'User-Agent': 'BigMap-Python-Client/1.0',
            'Accept': 'application/json'
        })
        
        self._species_functions = None
        
    def _rate_limited_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make a rate-limited request with proper error handling."""
        # Implement rate limiting
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            print_info(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        # Set timeout if not provided
        if 'timeout' not in kwargs:
            kwargs['timeout'] = self.timeout
            
        try:
            response = self.session.request(method, url, **kwargs)
            self._last_request_time = time.time()
            
            # Handle rate limiting responses
            if response.status_code == 429:
                retry_after = response.headers.get('Retry-After')
                if retry_after:
                    sleep_time = int(retry_after)
                    print_warning(f"Rate limited by server. Waiting {sleep_time}s...")
                    time.sleep(sleep_time)
                    # Retry once after rate limit
                    response = self.session.request(method, url, **kwargs)
                    self._last_request_time = time.time()
            
            return response
            
        except requests.exceptions.ConnectionError as e:
            print_error(f"Connection error: {e}")
            raise APIConnectionError(
                f"Connection error to FIA BIGMAP service",
                url=url,
                original_error=e
            )
        except requests.exceptions.Timeout as e:
            print_error(f"Request timeout after {self.timeout}s: {e}")
            raise APIConnectionError(
                f"Request timeout after {self.timeout}s",
                url=url,
                original_error=e
            )
        except requests.exceptions.RequestException as e:
            print_error(f"Request failed: {e}")
            raise APIConnectionError(
                f"Request to FIA BIGMAP service failed",
                url=url,
                original_error=e
            )
        
    def get_service_info(self) -> Dict:
        """Get basic service information."""
        try:
            print_info("Fetching service information...")
            response = self._rate_limited_request("GET", f"{self.base_url}?f=json")
            response.raise_for_status()
            result = response.json()
            print_success("Successfully retrieved service information")
            return result
        except requests.RequestException as e:
            print_error(f"Failed to get service info: {e}")
            raise APIConnectionError(
                "Failed to get service info from FIA BIGMAP",
                url=self.base_url,
                original_error=e
            )
    
    def get_species_functions(self) -> List[Dict]:
        """Get all available species raster functions."""
        if self._species_functions is None:
            info = self.get_service_info()
            if 'rasterFunctionInfos' in info:
                self._species_functions = info['rasterFunctionInfos']
                print_success(f"Found {len(self._species_functions)} raster functions")
            else:
                self._species_functions = []
                print_warning("No raster functions found in service info")
        return self._species_functions
    
    def list_available_species(self) -> List[Dict]:
        """Get list of all available species with codes and names."""
        functions = self.get_species_functions()
        species_list = []
        
        for func in functions:
            name = func.get('name', '')
            description = func.get('description', '')
            
            # Parse species code from function name
            if name.startswith('SPCD_') and name != 'SPCD_0000_TOTAL':
                parts = name.split('_')
                if len(parts) >= 2:
                    species_code = parts[1]
                    species_name = description
                    genus_species = '_'.join(parts[2:]) if len(parts) > 2 else ''
                    
                    species_list.append({
                        'species_code': species_code,
                        'common_name': species_name,
                        'scientific_name': genus_species.replace('_', ' '),
                        'function_name': name
                    })
        
        return sorted(species_list, key=lambda x: x['species_code'])
    
    def export_species_raster(
        self, 
        species_code: str,
        bbox: Tuple[float, float, float, float],
        output_path: Optional[Path] = None,
        pixel_size: float = 30.0,
        format: str = "tiff",
        bbox_srs: Union[str, int] = "102100",
        output_srs: Union[str, int] = "102100"
    ) -> Union[Path, np.ndarray]:
        """
        Export species biomass raster for a given bounding box.
        
        Args:
            species_code: FIA species code (e.g., "0131" for Loblolly Pine)
            bbox: Bounding box as (xmin, ymin, xmax, ymax)
            output_path: Path to save the raster file (optional)
            pixel_size: Pixel size in the units of output_srs
            format: Output format ("tiff", "png", "jpg")
            bbox_srs: Spatial reference of the bbox (WKID or "102100" for Web Mercator)
            output_srs: Output spatial reference (WKID or "102100" for Web Mercator, "2256" for Montana State Plane)
            
        Returns:
            Path to saved file or numpy array if no output_path
        """
        # Find the function name for this species
        function_name = self._get_function_name(species_code)
        if not function_name:
            print_error(f"Species code {species_code} not found")
            raise SpeciesNotFound(
                f"Species code {species_code} not found in FIA BIGMAP service",
                species_code=species_code
            )
        
        # Prepare export parameters
        params = {
            'f': 'json',
            'bbox': f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
            'bboxSR': str(bbox_srs),  # Input bbox spatial reference
            'imageSR': str(output_srs),  # Output spatial reference
            'format': format,
            'pixelType': 'F32',
            'renderingRule': json.dumps({
                'rasterFunction': function_name
            }),
            'size': self._calculate_image_size(bbox, pixel_size)
        }
        
        try:
            print_info(f"Exporting {function_name} for bbox {bbox}")
            
            # Make export request
            response = self._rate_limited_request("GET", f"{self.base_url}/exportImage", params=params)
            response.raise_for_status()
            
            result = response.json()
            
            if 'href' in result:
                # Download the actual raster data
                raster_response = self._rate_limited_request("GET", result['href'])
                raster_response.raise_for_status()
                
                if output_path:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, 'wb') as f:
                        f.write(raster_response.content)
                    print_success(f"Exported raster to {output_path}")
                    return output_path
                else:
                    # Return as numpy array
                    with MemoryFile(raster_response.content) as memfile:
                        with memfile.open() as dataset:
                            return dataset.read(1)
            else:
                print_error(f"Export failed: {result}")
                raise DownloadError(
                    f"Export failed for species {species_code}: {result}",
                    species_code=species_code,
                    output_path=str(output_path) if output_path else None
                )

        except requests.RequestException as e:
            print_error(f"Failed to export raster: {e}")
            raise DownloadError(
                f"Failed to export raster for species {species_code}",
                species_code=species_code,
                output_path=str(output_path) if output_path else None,
                original_error=e
            )
    
    def get_species_statistics(self, species_code: str) -> Dict:
        """Get statistics for a species across the entire dataset."""
        function_name = self._get_function_name(species_code)
        if not function_name:
            raise SpeciesNotFound(
                f"Species code {species_code} not found",
                species_code=species_code
            )
        
        params = {
            'f': 'json',
            'renderingRule': json.dumps({
                'rasterFunction': function_name
            })
        }
        
        try:
            response = self._rate_limited_request("GET", f"{self.base_url}/computeStatistics", params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print_error(f"Failed to get statistics: {e}")
            raise APIConnectionError(
                f"Failed to get statistics for species {species_code}",
                url=f"{self.base_url}/computeStatistics",
                original_error=e
            )
    
    def identify_pixel_value(
        self, 
        species_code: str, 
        x: float, 
        y: float, 
        spatial_ref: str = "102100"
    ) -> float:
        """
        Get biomass value for a species at a specific coordinate.
        
        Args:
            species_code: FIA species code
            x: X coordinate
            y: Y coordinate
            spatial_ref: Spatial reference system (default: Web Mercator)
            
        Returns:
            Biomass value at the location
        """
        function_name = self._get_function_name(species_code)
        if not function_name:
            raise SpeciesNotFound(
                f"Species code {species_code} not found",
                species_code=species_code
            )

        params = {
            'f': 'json',
            'geometry': f"{x},{y}",
            'geometryType': 'esriGeometryPoint',
            'sr': spatial_ref,
            'renderingRule': json.dumps({
                'rasterFunction': function_name
            })
        }
        
        try:
            response = self._rate_limited_request("GET", f"{self.base_url}/identify", params=params)
            response.raise_for_status()
            result = response.json()
            
            if 'value' in result:
                value = result['value']
                if value == 'NoData' or value is None:
                    return 0.0  # No biomass at this location
                return float(value)
            return None
            
        except requests.RequestException as e:
            print_error(f"Failed to identify pixel: {e}")
            raise APIConnectionError(
                f"Failed to identify pixel value for species {species_code}",
                url=f"{self.base_url}/identify",
                original_error=e
            )
    
    def export_total_biomass_raster(
        self,
        bbox: Tuple[float, float, float, float],
        output_path: Optional[Path] = None,
        pixel_size: float = 30.0,
        format: str = "tiff",
        bbox_srs: Union[str, int] = "102100",
        output_srs: Union[str, int] = "102100"
    ) -> Union[Path, np.ndarray]:
        """
        Export total biomass raster for a given bounding box.
        
        Args:
            bbox: Bounding box as (xmin, ymin, xmax, ymax)
            output_path: Path to save the raster file (optional)
            pixel_size: Pixel size in the units of output_srs
            format: Output format ("tiff", "png", "jpg")
            bbox_srs: Spatial reference of the bbox
            output_srs: Output spatial reference
            
        Returns:
            Path to saved file or numpy array if no output_path
        """
        # For total biomass, use no rendering rule
        params = {
            'f': 'json',
            'bbox': f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
            'bboxSR': str(bbox_srs),
            'imageSR': str(output_srs),
            'format': format,
            'pixelType': 'F32',
            'size': self._calculate_image_size(bbox, pixel_size)
        }
        
        try:
            print_info(f"Exporting total biomass for bbox {bbox}")
            
            # Make export request
            response = self._rate_limited_request("GET", f"{self.base_url}/exportImage", params=params)
            response.raise_for_status()
            
            result = response.json()
            
            if 'href' in result:
                # Download the actual raster data
                raster_response = self._rate_limited_request("GET", result['href'])
                raster_response.raise_for_status()
                
                if output_path:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, 'wb') as f:
                        f.write(raster_response.content)
                    print_success(f"Exported total biomass to {output_path}")
                    return output_path
                else:
                    # Return as numpy array
                    with MemoryFile(raster_response.content) as memfile:
                        with memfile.open() as dataset:
                            return dataset.read(1)
            else:
                print_error(f"Export failed: {result}")
                raise DownloadError(
                    f"Export failed for total biomass: {result}",
                    output_path=str(output_path) if output_path else None
                )

        except requests.RequestException as e:
            print_error(f"Failed to export total biomass: {e}")
            raise DownloadError(
                "Failed to export total biomass raster",
                output_path=str(output_path) if output_path else None,
                original_error=e
            )
    
    def batch_export_location_species(
        self, 
        bbox: Tuple[float, float, float, float],
        output_dir: Path,
        species_codes: Optional[List[str]] = None,
        location_name: str = "location",
        bbox_srs: Union[str, int] = "102100",
        output_srs: Union[str, int] = "102100"
    ) -> List[Path]:
        """
        Batch export multiple species for any geographic location.
        
        Args:
            bbox: Bounding box in the specified CRS
            output_dir: Directory to save raster files
            species_codes: List of species codes to export (optional)
            location_name: Name prefix for output files
            bbox_srs: Spatial reference of the bbox
            output_srs: Output spatial reference
            
        Returns:
            List of paths to exported files
        """
        if species_codes is None:
            # Get all available species
            all_species = self.list_available_species()
            species_codes = [s['species_code'] for s in all_species]
        
        output_dir.mkdir(parents=True, exist_ok=True)
        exported_files = []
        
        with Progress() as progress:
            task = progress.add_task("Exporting species...", total=len(species_codes))
            
            for species_code in species_codes:
                output_file = output_dir / f"{location_name}_species_{species_code}.tif"
                
                try:
                    result = self.export_species_raster(
                        species_code=species_code,
                        bbox=bbox,
                        output_path=output_file,
                        bbox_srs=bbox_srs,
                        output_srs=output_srs
                    )
                    
                    if result:
                        exported_files.append(result)
                        
                except Exception as e:
                    print_warning(f"Failed to export species {species_code}: {e}")
                
                progress.update(task, advance=1)
        
        print_success(f"Exported {len(exported_files)} species rasters to {output_dir}")
        return exported_files
    
    def _get_function_name(self, species_code: str) -> Optional[str]:
        """Get the raster function name for a species code."""
        functions = self.get_species_functions()
        
        for func in functions:
            name = func.get('name', '')
            if name.startswith(f'SPCD_{species_code}_'):
                return name
        
        return None
    
    def _calculate_image_size(
        self, 
        bbox: Tuple[float, float, float, float], 
        pixel_size: float
    ) -> str:
        """Calculate image size based on bbox and pixel size."""
        width = int((bbox[2] - bbox[0]) / pixel_size)
        height = int((bbox[3] - bbox[1]) / pixel_size)
        
        # Limit to service maximums
        max_width = 15000
        max_height = 4100
        
        if width > max_width:
            width = max_width
        if height > max_height:
            height = max_height
            
        return f"{width},{height}" 