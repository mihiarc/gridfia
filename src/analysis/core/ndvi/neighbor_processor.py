"""Module for processing NDVI data specifically for neighbor properties."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import box

from ..property_matching.matcher import PropertyMatcher
from .processor import NDVIProcessor

logger = logging.getLogger(__name__)

class NeighborSelector:
    """Responsible for selecting and validating neighbor properties."""
    
    def __init__(self, matches_file: Path, processed_dir: Path):
        """Initialize the neighbor selector.
        
        Args:
            matches_file: Path to the property matches file
            processed_dir: Directory containing processed property data
        """
        self.matches_file = matches_file
        self.processed_dir = processed_dir
    
    def get_neighbors_within_bounds(self, bounds_poly: box, ndvi_crs: str) -> gpd.GeoDataFrame:
        """Get all neighbor properties within the given bounds.
        
        Args:
            bounds_poly: Shapely box representing the bounds
            ndvi_crs: CRS of the NDVI data
            
        Returns:
            GeoDataFrame of neighbor properties within bounds
        """
        # Load matches
        matches = pd.read_csv(self.matches_file)
        
        # Load neighbor parcels
        neighbor_ids = matches['neighbor_id'].unique()
        parcels = pd.read_parquet(
            self.processed_dir / "parcels_processed.parquet",
            filters=[('property_id', 'in', neighbor_ids.tolist())]
        )
        
        # Convert to GeoDataFrame and filter by bounds
        neighbors_gdf = gpd.GeoDataFrame(parcels, geometry='geometry', crs="EPSG:2264")
        neighbors_gdf = neighbors_gdf.to_crs(ndvi_crs)
        neighbors_in_bounds = neighbors_gdf[neighbors_gdf.intersects(bounds_poly)]
        
        logger.info(f"Found {len(neighbors_in_bounds)} neighbor properties within NDVI coverage")
        return neighbors_in_bounds

class NDVIExtractor:
    """Responsible for extracting NDVI values from raster data."""
    
    def __init__(self, ndvi_dir: Path):
        """Initialize the NDVI extractor.
        
        Args:
            ndvi_dir: Directory containing NDVI raster files
        """
        self.ndvi_dir = ndvi_dir
    
    def get_ndvi_bounds(self) -> Tuple[box, str]:
        """Get the combined bounds of all NDVI raster files.
        
        Returns:
            Tuple of (bounds_polygon, crs)
        """
        ndvi_files = []
        for part in ['p1', 'p2']:
            pattern = f"ndvi_NAIP_Vance_County_2018-{part}.tif"
            matches = list(self.ndvi_dir.glob(pattern))
            ndvi_files.extend(matches)
        
        if not ndvi_files:
            raise FileNotFoundError(f"No NDVI files found in {self.ndvi_dir}")
        
        bounds_list = []
        ndvi_crs = None
        
        for ndvi_file in ndvi_files:
            with rasterio.open(ndvi_file) as src:
                bounds_list.append(src.bounds)
                if ndvi_crs is None:
                    ndvi_crs = src.crs
                elif ndvi_crs != src.crs:
                    raise ValueError(f"Inconsistent CRS found: {ndvi_file}")
        
        combined_bounds = rasterio.coords.BoundingBox(
            left=min(b.left for b in bounds_list),
            bottom=min(b.bottom for b in bounds_list),
            right=max(b.right for b in bounds_list),
            top=max(b.top for b in bounds_list)
        )
        
        bounds_poly = box(
            combined_bounds.left, 
            combined_bounds.bottom,
            combined_bounds.right, 
            combined_bounds.top
        )
        
        return bounds_poly, ndvi_crs

class ResultsManager:
    """Responsible for managing and saving NDVI results."""
    
    def __init__(self, output_dir: Path):
        """Initialize the results manager.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_results(self, results_df: pd.DataFrame, prefix: str = "neighbors") -> Path:
        """Save NDVI results to file.
        
        Args:
            results_df: DataFrame containing NDVI results
            prefix: Prefix for the output filename
            
        Returns:
            Path to saved file
        """
        output_file = self.output_dir / f"{prefix}_ndvi_results.parquet"
        results_df.to_parquet(output_file)
        logger.info(f"Saved NDVI results to {output_file}")
        return output_file

class NeighborNDVIProcessor:
    """Main class for processing NDVI data for neighbor properties."""
    
    def __init__(
        self,
        ndvi_dir: Path,
        output_dir: Path,
        matches_file: Path,
        processed_dir: Path,
        n_workers: Optional[int] = None
    ):
        """Initialize the neighbor NDVI processor.
        
        Args:
            ndvi_dir: Directory containing NDVI files
            output_dir: Directory to save results
            matches_file: Path to property matches file
            processed_dir: Directory containing processed property data
            n_workers: Number of worker processes
        """
        self.ndvi_extractor = NDVIExtractor(ndvi_dir)
        self.neighbor_selector = NeighborSelector(matches_file, processed_dir)
        self.results_manager = ResultsManager(output_dir)
        self.n_workers = n_workers
    
    def process(self) -> Path:
        """Process NDVI data for all neighbor properties.
        
        Returns:
            Path to results file
        """
        logger.info("Starting neighbor NDVI processing")
        
        # Get NDVI bounds
        bounds_poly, ndvi_crs = self.ndvi_extractor.get_ndvi_bounds()
        logger.info(f"NDVI data CRS: {ndvi_crs}")
        
        # Get neighbors within bounds
        neighbors_gdf = self.neighbor_selector.get_neighbors_within_bounds(bounds_poly, ndvi_crs)
        
        if len(neighbors_gdf) == 0:
            logger.warning("No neighbor properties found within NDVI bounds")
            return None
        
        # Process NDVI for neighbors
        ndvi_processor = NDVIProcessor(self.n_workers)
        results_df = ndvi_processor.process_properties(neighbors_gdf)
        
        # Save and return results
        return self.results_manager.save_results(results_df) 