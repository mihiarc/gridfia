import os
import glob
from pathlib import Path
import rasterio
from osgeo import gdal
import tempfile
import logging
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MosaicCreator:
    """Handles the creation of NDVI mosaics from multiple input files."""
    
    def __init__(self, ndvi_dir: str, output_dir: str):
        """
        Initialize the MosaicCreator.
        
        Args:
            ndvi_dir: Directory containing NDVI files
            output_dir: Directory to save mosaic outputs
        """
        self.ndvi_dir = Path(ndvi_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = None
        
    def _setup_temp_dir(self):
        """Create temporary directory for mosaic processing."""
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix='ndvi_mosaic_')
            logger.info(f"Created temporary directory: {self.temp_dir}")
    
    def _cleanup_temp_dir(self):
        """Clean up temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            self.temp_dir = None
    
    def group_files_by_year(self) -> Dict[str, List[str]]:
        """
        Group NDVI files by year.
        
        Returns:
            Dictionary mapping years to lists of file paths
        """
        files_by_year = {}
        pattern = str(self.ndvi_dir / "ndvi_NAIP_*_County_*.tif")
        
        for file_path in glob.glob(pattern):
            # Extract year from filename
            filename = os.path.basename(file_path)
            year = filename.split('_')[-1].split('-')[0]
            
            if year not in files_by_year:
                files_by_year[year] = []
            files_by_year[year].append(file_path)
        
        return files_by_year
    
    def create_vrt(self, files: List[str], year: str) -> str:
        """
        Create a VRT file for the given input files.
        
        Args:
            files: List of input file paths
            year: Year for the mosaic
            
        Returns:
            Path to created VRT file
        """
        self._setup_temp_dir()
        vrt_path = os.path.join(self.temp_dir, f"ndvi_{year}.vrt")
        
        vrt_options = gdal.BuildVRTOptions(resampleAlg='nearest')
        gdal.BuildVRT(vrt_path, files, options=vrt_options)
        
        logger.info(f"Created VRT for year {year}: {vrt_path}")
        return vrt_path
    
    def create_mosaic(self, year: str, files: List[str], save_final: bool = False) -> str:
        """
        Create a mosaic for the given year.
        
        Args:
            year: Year to create mosaic for
            files: List of input files
            save_final: Whether to save the final mosaic or keep as VRT
            
        Returns:
            Path to mosaic (VRT or TIF)
        """
        vrt_path = self.create_vrt(files, year)
        
        if save_final:
            output_path = str(self.output_dir / f"ndvi_{year}_mosaic.tif")
            
            # Translate VRT to GeoTIFF
            translate_options = gdal.TranslateOptions(format='GTiff', 
                                                    creationOptions=['COMPRESS=LZW'])
            gdal.Translate(output_path, vrt_path, options=translate_options)
            
            logger.info(f"Created final mosaic for year {year}: {output_path}")
            return output_path
        
        return vrt_path
    
    def create_all_mosaics(self, save_final: bool = False) -> Dict[str, str]:
        """
        Create mosaics for all years.
        
        Args:
            save_final: Whether to save final GeoTIFFs
            
        Returns:
            Dictionary mapping years to mosaic paths
        """
        files_by_year = self.group_files_by_year()
        mosaic_paths = {}
        
        for year, files in files_by_year.items():
            if files:
                mosaic_paths[year] = self.create_mosaic(year, files, save_final)
                logger.info(f"Completed mosaic for year {year}")
        
        return mosaic_paths
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self._cleanup_temp_dir() 