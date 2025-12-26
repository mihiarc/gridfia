"""
REST API client for FIA BIGMAP ImageServer access.
"""

import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import threading
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import rasterio
from rasterio.io import MemoryFile
import numpy as np
from rich.console import Console
from rich.progress import Progress, track

from ..console import print_info, print_success, print_error, print_warning
from ..exceptions import APIConnectionError, SpeciesNotFound, DownloadError, CircuitBreakerOpen

console = Console()


class CircuitBreakerState(Enum):
    """States for the circuit breaker pattern."""
    CLOSED = "closed"      # Normal operation, requests allowed
    OPEN = "open"          # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing recovery, limited requests allowed


class CircuitBreaker:
    """
    Thread-safe circuit breaker implementation for protecting against cascading failures.

    The circuit breaker has three states:
    - CLOSED: Normal operation. All requests pass through. Failures are counted.
    - OPEN: Failure threshold exceeded. All requests are blocked immediately.
    - HALF_OPEN: After recovery timeout, one test request is allowed to check recovery.

    State transitions:
    - CLOSED -> OPEN: When failure_count >= failure_threshold
    - OPEN -> HALF_OPEN: When recovery_timeout has elapsed since last failure
    - HALF_OPEN -> CLOSED: When a test request succeeds
    - HALF_OPEN -> OPEN: When a test request fails

    Examples
    --------
    >>> breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
    >>> if breaker.can_execute():
    ...     try:
    ...         result = make_request()
    ...         breaker.record_success()
    ...     except Exception:
    ...         breaker.record_failure()
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        name: str = "default"
    ):
        """
        Initialize the circuit breaker.

        Parameters
        ----------
        failure_threshold : int
            Number of consecutive failures before opening the circuit.
            Default is 5.
        recovery_timeout : float
            Time in seconds to wait before attempting recovery (HALF_OPEN state).
            Default is 60.0 seconds.
        name : str
            Name for this circuit breaker instance (used in logging).
        """
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._name = name

        # State management
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._last_state_change_time = time.time()

        # Thread safety
        self._lock = threading.RLock()

    @property
    def state(self) -> CircuitBreakerState:
        """Get the current circuit breaker state."""
        with self._lock:
            self._check_state_transition()
            return self._state

    @property
    def failure_count(self) -> int:
        """Get the current failure count."""
        with self._lock:
            return self._failure_count

    @property
    def failure_threshold(self) -> int:
        """Get the failure threshold."""
        return self._failure_threshold

    @property
    def recovery_timeout(self) -> float:
        """Get the recovery timeout in seconds."""
        return self._recovery_timeout

    @property
    def time_until_recovery(self) -> Optional[float]:
        """Get the time remaining until recovery attempt, or None if not in OPEN state."""
        with self._lock:
            if self._state != CircuitBreakerState.OPEN or self._last_failure_time is None:
                return None
            elapsed = time.time() - self._last_failure_time
            remaining = self._recovery_timeout - elapsed
            return max(0.0, remaining)

    def _check_state_transition(self) -> None:
        """
        Check if state should transition (must be called while holding lock).

        Handles automatic transition from OPEN to HALF_OPEN when recovery timeout elapses.
        """
        if self._state == CircuitBreakerState.OPEN and self._last_failure_time is not None:
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self._recovery_timeout:
                self._transition_to(CircuitBreakerState.HALF_OPEN)

    def _transition_to(self, new_state: CircuitBreakerState) -> None:
        """
        Transition to a new state (must be called while holding lock).

        Parameters
        ----------
        new_state : CircuitBreakerState
            The state to transition to.
        """
        old_state = self._state
        self._state = new_state
        self._last_state_change_time = time.time()

        print_info(
            f"Circuit breaker '{self._name}' state transition: "
            f"{old_state.value} -> {new_state.value}"
        )

    def can_execute(self) -> bool:
        """
        Check if a request can be executed.

        Returns
        -------
        bool
            True if the request is allowed, False otherwise.

        Notes
        -----
        In CLOSED state, always returns True.
        In OPEN state, returns False until recovery timeout elapses.
        In HALF_OPEN state, returns True to allow a test request.
        """
        with self._lock:
            self._check_state_transition()

            if self._state == CircuitBreakerState.CLOSED:
                return True
            elif self._state == CircuitBreakerState.OPEN:
                return False
            else:  # HALF_OPEN
                return True

    def record_success(self) -> None:
        """
        Record a successful request.

        In CLOSED state: resets failure count.
        In HALF_OPEN state: transitions to CLOSED.
        """
        with self._lock:
            if self._state == CircuitBreakerState.HALF_OPEN:
                print_success(
                    f"Circuit breaker '{self._name}' recovery successful, closing circuit"
                )
                self._transition_to(CircuitBreakerState.CLOSED)

            self._failure_count = 0

    def record_failure(self) -> None:
        """
        Record a failed request.

        In CLOSED state: increments failure count, may transition to OPEN.
        In HALF_OPEN state: transitions back to OPEN.
        """
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitBreakerState.HALF_OPEN:
                print_warning(
                    f"Circuit breaker '{self._name}' recovery failed, reopening circuit"
                )
                self._transition_to(CircuitBreakerState.OPEN)
            elif self._state == CircuitBreakerState.CLOSED:
                if self._failure_count >= self._failure_threshold:
                    print_error(
                        f"Circuit breaker '{self._name}' opened after "
                        f"{self._failure_count} consecutive failures"
                    )
                    self._transition_to(CircuitBreakerState.OPEN)

    def reset(self) -> None:
        """
        Reset the circuit breaker to its initial CLOSED state.

        This clears all failure counts and resets the state to CLOSED.
        Useful for manual recovery or testing purposes.
        """
        with self._lock:
            old_state = self._state
            self._state = CircuitBreakerState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None
            self._last_state_change_time = time.time()

            if old_state != CircuitBreakerState.CLOSED:
                print_info(f"Circuit breaker '{self._name}' manually reset to CLOSED")

    def get_status(self) -> Dict:
        """
        Get the current status of the circuit breaker.

        Returns
        -------
        dict
            A dictionary containing the current state, failure count,
            threshold, and time until recovery (if applicable).
        """
        with self._lock:
            self._check_state_transition()
            return {
                "name": self._name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "failure_threshold": self._failure_threshold,
                "recovery_timeout": self._recovery_timeout,
                "time_until_recovery": self.time_until_recovery,
                "last_failure_time": self._last_failure_time,
                "last_state_change_time": self._last_state_change_time
            }


class BigMapRestClient:
    """Client for accessing FIA BIGMAP ImageServer REST API with proper retry, rate limiting, and circuit breaker."""

    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        timeout: int = 30,
        rate_limit_delay: float = 0.5,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 60.0,
        circuit_breaker_enabled: bool = True
    ):
        """
        Initialize the REST client with retry, rate limiting, and circuit breaker configuration.

        Args:
            max_retries: Maximum number of retries for failed requests
            backoff_factor: Backoff factor for retry delays
            timeout: Request timeout in seconds
            rate_limit_delay: Delay between requests in seconds
            circuit_breaker_threshold: Number of consecutive failures before opening circuit
            circuit_breaker_timeout: Time in seconds to wait before recovery attempt
            circuit_breaker_enabled: Whether to enable circuit breaker protection
        """
        self.base_url = "https://di-usfsdata.img.arcgis.com/arcgis/rest/services/FIA_BIGMAP_2018_Tree_Species_Aboveground_Biomass/ImageServer"
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time = 0

        # Circuit breaker configuration
        self._circuit_breaker_enabled = circuit_breaker_enabled
        self._circuit_breaker: Optional[CircuitBreaker] = None
        if circuit_breaker_enabled:
            self._circuit_breaker = CircuitBreaker(
                failure_threshold=circuit_breaker_threshold,
                recovery_timeout=circuit_breaker_timeout,
                name="FIA_BIGMAP"
            )
        
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

    @property
    def circuit_breaker(self) -> Optional[CircuitBreaker]:
        """Get the circuit breaker instance, if enabled."""
        return self._circuit_breaker

    def get_circuit_breaker_status(self) -> Optional[Dict]:
        """
        Get the current status of the circuit breaker.

        Returns
        -------
        dict or None
            Circuit breaker status if enabled, None otherwise.
        """
        if self._circuit_breaker is not None:
            return self._circuit_breaker.get_status()
        return None

    def reset_circuit_breaker(self) -> None:
        """
        Manually reset the circuit breaker to CLOSED state.

        This can be used to force recovery after a service has been restored.
        """
        if self._circuit_breaker is not None:
            self._circuit_breaker.reset()

    def _rate_limited_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make a rate-limited request with circuit breaker protection and proper error handling."""
        # Check circuit breaker first
        if self._circuit_breaker_enabled and self._circuit_breaker is not None:
            if not self._circuit_breaker.can_execute():
                retry_after = self._circuit_breaker.time_until_recovery
                raise CircuitBreakerOpen(
                    "Circuit breaker is OPEN - FIA BIGMAP service is currently unavailable",
                    failure_count=self._circuit_breaker.failure_count,
                    failure_threshold=self._circuit_breaker.failure_threshold,
                    retry_after=retry_after,
                    last_failure_time=self._circuit_breaker._last_failure_time
                )

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

            # Check for server errors that should trigger circuit breaker
            if response.status_code >= 500:
                if self._circuit_breaker_enabled and self._circuit_breaker is not None:
                    self._circuit_breaker.record_failure()
            else:
                # Record success for non-error responses
                if self._circuit_breaker_enabled and self._circuit_breaker is not None:
                    self._circuit_breaker.record_success()

            return response

        except requests.exceptions.ConnectionError as e:
            # Record failure in circuit breaker
            if self._circuit_breaker_enabled and self._circuit_breaker is not None:
                self._circuit_breaker.record_failure()
            print_error(f"Connection error: {e}")
            raise APIConnectionError(
                f"Connection error to FIA BIGMAP service",
                url=url,
                original_error=e
            )
        except requests.exceptions.Timeout as e:
            # Record failure in circuit breaker
            if self._circuit_breaker_enabled and self._circuit_breaker is not None:
                self._circuit_breaker.record_failure()
            print_error(f"Request timeout after {self.timeout}s: {e}")
            raise APIConnectionError(
                f"Request timeout after {self.timeout}s",
                url=url,
                original_error=e
            )
        except requests.exceptions.RequestException as e:
            # Record failure in circuit breaker
            if self._circuit_breaker_enabled and self._circuit_breaker is not None:
                self._circuit_breaker.record_failure()
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