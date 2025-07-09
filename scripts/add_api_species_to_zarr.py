#!/usr/bin/env python3
"""
Download a species from REST API and add it to the zarr with proper spatial alignment.

This script:
1. Gets the spatial extent and CRS from the existing zarr
2. Downloads a species raster from the REST API
3. Clips and resamples to match zarr dimensions exactly
4. Adds the species to the zarr file
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.io import MemoryFile
import zarr
from pathlib import Path
import tempfile
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import time

from bigmap.api import BigMapRestClient
from bigmap.utils.create_nc_biomass_zarr import append_species_to_zarr
from bigmap.console import print_info, print_success, print_error, print_warning

console = Console()

class APISpeciesProcessor:
    """Handles downloading and processing species from REST API to zarr."""
    
    def __init__(self, zarr_path: str = "./output/nc_biomass_expandable.zarr"):
        self.zarr_path = zarr_path
        self.client = BigMapRestClient()
        self.zarr_spatial_info = None
        
        # Load zarr spatial information
        self._load_zarr_spatial_info()
        
    def _load_zarr_spatial_info(self):
        """Load spatial reference information from the zarr."""
        print_info("Loading spatial information from zarr...")
        
        zarr_array = zarr.open_array(self.zarr_path, mode='r')
        
        # Extract spatial info from zarr attributes
        self.zarr_spatial_info = {
            'height': zarr_array.attrs['height'],
            'width': zarr_array.attrs['width'],
            'transform': zarr_array.attrs['transform'],
            'bounds': zarr_array.attrs['bounds'],
            'crs': zarr_array.attrs['crs'],
            'shape': (zarr_array.attrs['height'], zarr_array.attrs['width'])
        }
        
        print_success(f"Zarr spatial info loaded: {self.zarr_spatial_info['shape']} @ {self.zarr_spatial_info['crs']}")
        
    def get_nc_bbox_web_mercator(self):
        """Get North Carolina bounding box in Web Mercator for API requests."""
        # Convert zarr bounds to Web Mercator (EPSG:3857)
        from rasterio.warp import transform_bounds
        
        # zarr bounds are in zarr CRS, convert to Web Mercator
        bounds = self.zarr_spatial_info['bounds']
        zarr_crs = self.zarr_spatial_info['crs']
        
        # Transform to Web Mercator
        web_mercator_bounds = transform_bounds(
            zarr_crs, 'EPSG:3857', 
            bounds[0], bounds[1], bounds[2], bounds[3]
        )
        
        print_info(f"NC bounds in Web Mercator: {web_mercator_bounds}")
        return web_mercator_bounds
        
    def download_species_raster(self, species_code: str) -> np.ndarray:
        """Download species raster from REST API."""
        print_info(f"Downloading species {species_code} from REST API...")
        
        # Get bounding box in Web Mercator
        bbox = self.get_nc_bbox_web_mercator()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Downloading {species_code}...", total=None)
            
            # Download raster data
            raster_data = self.client.export_species_raster(
                species_code=species_code,
                bbox=bbox,
                output_path=None,  # Return as numpy array
                pixel_size=30.0,
                format="tiff"
            )
            
            progress.update(task, completed=True)
            
        if raster_data is None:
            raise ValueError(f"Failed to download species {species_code}")
            
        print_success(f"Downloaded raster shape: {raster_data.shape}")
        return raster_data
        
    def align_raster_to_zarr(self, raster_data: np.ndarray, raster_transform, raster_crs: str) -> np.ndarray:
        """Align downloaded raster to match zarr spatial grid exactly."""
        print_info("Aligning raster to zarr spatial grid...")
        
        target_height = self.zarr_spatial_info['height']
        target_width = self.zarr_spatial_info['width']
        target_transform = rasterio.transform.from_bounds(
            *self.zarr_spatial_info['bounds'], 
            target_width, 
            target_height
        )
        target_crs = self.zarr_spatial_info['crs']
        
        print_info(f"Source: {raster_data.shape} in {raster_crs}")
        print_info(f"Target: ({target_height}, {target_width}) in {target_crs}")
        
        # Create output array with target dimensions
        aligned_data = np.zeros((target_height, target_width), dtype=np.float32)
        
        # Reproject to match zarr grid exactly
        reproject(
            source=raster_data,
            destination=aligned_data,
            src_transform=raster_transform,
            src_crs=raster_crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,  # Use bilinear for biomass data
            src_nodata=0,
            dst_nodata=0
        )
        
        # Report alignment results
        valid_pixels = np.count_nonzero(aligned_data)
        total_pixels = aligned_data.size
        coverage_pct = (valid_pixels / total_pixels) * 100
        
        print_success(f"Aligned raster: {aligned_data.shape}")
        print_info(f"Coverage: {coverage_pct:.2f}% ({valid_pixels:,} pixels)")
        if valid_pixels > 0:
            print_info(f"Biomass range: {aligned_data[aligned_data > 0].min():.2f} - {aligned_data.max():.2f}")
            print_info(f"Mean biomass: {aligned_data[aligned_data > 0].mean():.2f}")
        
        return aligned_data
        
    def add_species_to_zarr(self, species_code: str, aligned_data: np.ndarray, species_name: str):
        """Add the aligned species data to the zarr file."""
        print_info(f"Adding {species_code} ({species_name}) to zarr...")
        
        # Save aligned data to temporary file for append function
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_file:
            temp_path = tmp_file.name
            
        try:
            # Write aligned data to temporary GeoTIFF
            with rasterio.open(
                temp_path, 'w',
                driver='GTiff',
                height=aligned_data.shape[0],
                width=aligned_data.shape[1],
                count=1,
                dtype=aligned_data.dtype,
                crs=self.zarr_spatial_info['crs'],
                transform=rasterio.transform.from_bounds(
                    *self.zarr_spatial_info['bounds'],
                    self.zarr_spatial_info['width'],
                    self.zarr_spatial_info['height']
                ),
                nodata=0
            ) as dst:
                dst.write(aligned_data, 1)
            
            # Use existing append function
            append_species_to_zarr(
                zarr_path=self.zarr_path,
                species_raster_path=temp_path,
                species_code=f"SPCD{species_code}",
                species_name=species_name
            )
            
            print_success(f"Successfully added {species_code} to zarr!")
            
        finally:
            # Clean up temporary file
            Path(temp_path).unlink(missing_ok=True)
            
    def process_species(self, species_code: str, species_name: str = None):
        """Complete workflow: download, align, and add species to zarr."""
        
        # Get species name if not provided
        if species_name is None:
            species_list = self.client.list_available_species()
            species_dict = {s['species_code']: s['common_name'] for s in species_list}
            species_name = species_dict.get(species_code, f"Species_{species_code}")
        
        console.print(f"\n[bold blue]Processing Species {species_code}[/bold blue]")
        console.print(f"Name: {species_name}")
        console.print(f"Target zarr: {self.zarr_path}")
        console.print()
        
        try:
            # Step 1: Download from API
            with console.status("[bold green]Downloading from REST API..."):
                # Download returns numpy array but we need the spatial info too
                # So we'll export to a temporary file first
                bbox = self.get_nc_bbox_web_mercator()
                
                with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_file:
                    download_path = tmp_file.name
                
                try:
                    result_path = self.client.export_species_raster(
                        species_code=species_code,
                        bbox=bbox,
                        output_path=Path(download_path),
                        pixel_size=30.0,
                        format="tiff"
                    )
                    
                    if not result_path:
                        raise ValueError("Download failed")
                    
                    # Read the downloaded raster with spatial info
                    with rasterio.open(download_path) as src:
                        raster_data = src.read(1)
                        raster_transform = src.transform
                        raster_crs = src.crs
                    
                finally:
                    Path(download_path).unlink(missing_ok=True)
            
            print_success("Download completed")
            
            # Step 2: Align to zarr grid
            aligned_data = self.align_raster_to_zarr(raster_data, raster_transform, raster_crs)
            
            # Step 3: Add to zarr
            self.add_species_to_zarr(species_code, aligned_data, species_name)
            
            # Show final status
            zarr_array = zarr.open_array(self.zarr_path, mode='r')
            final_shape = zarr_array.shape
            final_species_count = zarr_array.attrs['n_species']
            
            success_panel = Panel(
                f"[bold green]Success![/bold green]\n\n"
                f"Species {species_code} ({species_name}) added to zarr\n"
                f"Final zarr shape: {final_shape}\n"
                f"Total species: {final_species_count}",
                title="Processing Complete",
                border_style="green"
            )
            console.print(success_panel)
            
        except Exception as e:
            error_panel = Panel(
                f"[bold red]Error processing {species_code}:[/bold red]\n\n{str(e)}",
                title="Processing Failed",
                border_style="red"
            )
            console.print(error_panel)
            raise

def main():
    """Main function for interactive testing."""
    import sys
    
    if len(sys.argv) < 2:
        console.print("[bold red]Usage:[/bold red] python add_api_species_to_zarr.py <species_code> [species_name]")
        console.print("\nExample: python add_api_species_to_zarr.py 0131 'Loblolly Pine'")
        console.print("\nAvailable species codes can be found with: bigmap list-api-species")
        return
    
    species_code = sys.argv[1]
    species_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    processor = APISpeciesProcessor()
    processor.process_species(species_code, species_name)

if __name__ == "__main__":
    main() 