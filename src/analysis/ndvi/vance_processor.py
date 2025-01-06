"""
Vance County NDVI Processing Module
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import logging

from ..config.vance_config import (
    ANALYSIS_YEARS,
    BATCH_SIZE,
    MAX_WORKERS,
    MIN_VALID_PIXELS,
    MAX_INVALID_RATIO,
    NDVI_DIR
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VanceNDVIProcessor:
    """Handles NDVI processing for Vance County properties."""
    
    def __init__(self, properties_file: Path):
        """Initialize the NDVI processor.
        
        Args:
            properties_file: Path to filtered Vance properties
        """
        self.properties_file = properties_file
        self.ndvi_files = self._get_ndvi_files()
        
    def _get_ndvi_files(self) -> Dict[int, List[Path]]:
        """Get NDVI files for each analysis year.
        
        Returns:
            Dictionary mapping years to NDVI file paths
        """
        ndvi_files = {}
        for year in ANALYSIS_YEARS:
            files = list(NDVI_DIR.glob(f"*{year}*.tif"))
            if not files:
                raise FileNotFoundError(f"No NDVI files found for {year}")
            ndvi_files[year] = sorted(files)
        return ndvi_files
    
    def extract_ndvi_stats(
        self,
        geometry: gpd.GeoSeries,
        ndvi_file: Path
    ) -> Dict[str, float]:
        """Extract NDVI statistics for a single property.
        
        Args:
            geometry: Property geometry
            ndvi_file: Path to NDVI file
            
        Returns:
            Dictionary of NDVI statistics
        """
        with rasterio.open(ndvi_file) as src:
            try:
                # Mask NDVI data to property boundary
                ndvi_data, _ = mask(src, [geometry], crop=True)
                
                # Filter invalid pixels
                valid_mask = (ndvi_data != src.nodata) & (ndvi_data >= -1) & (ndvi_data <= 1)
                valid_pixels = ndvi_data[valid_mask]
                
                if len(valid_pixels) < MIN_VALID_PIXELS:
                    logger.warning(f"Insufficient valid pixels: {len(valid_pixels)}")
                    return None
                
                invalid_ratio = 1 - (len(valid_pixels) / ndvi_data.size)
                if invalid_ratio > MAX_INVALID_RATIO:
                    logger.warning(f"Too many invalid pixels: {invalid_ratio:.2%}")
                    return None
                
                return {
                    'mean': float(np.mean(valid_pixels)),
                    'std': float(np.std(valid_pixels)),
                    'min': float(np.min(valid_pixels)),
                    'max': float(np.max(valid_pixels)),
                    'pixel_count': int(len(valid_pixels)),
                    'invalid_ratio': float(invalid_ratio)
                }
                
            except Exception as e:
                logger.error(f"Error processing property: {e}")
                return None
    
    def process_property_batch(
        self,
        properties: gpd.GeoDataFrame
    ) -> pd.DataFrame:
        """Process NDVI statistics for a batch of properties.
        
        Args:
            properties: Batch of properties to process
            
        Returns:
            DataFrame of NDVI statistics
        """
        results = []
        for _, prop in properties.iterrows():
            prop_stats = {}
            for year in ANALYSIS_YEARS:
                # Process each NDVI file for the year
                year_stats = []
                for ndvi_file in self.ndvi_files[year]:
                    stats = self.extract_ndvi_stats(prop.geometry, ndvi_file)
                    if stats:
                        year_stats.append(stats)
                
                if year_stats:
                    # Average statistics across files
                    prop_stats[f'ndvi_{year}_mean'] = np.mean([s['mean'] for s in year_stats])
                    prop_stats[f'ndvi_{year}_std'] = np.mean([s['std'] for s in year_stats])
                    prop_stats[f'pixel_count_{year}'] = sum([s['pixel_count'] for s in year_stats])
            
            if prop_stats:
                prop_stats['property_id'] = prop.property_id
                results.append(prop_stats)
        
        return pd.DataFrame(results)
    
    def process_all_properties(self) -> pd.DataFrame:
        """Process NDVI statistics for all Vance County properties.
        
        Returns:
            DataFrame of NDVI statistics for all properties
        """
        logger.info("Loading Vance County properties...")
        properties = gpd.read_parquet(self.properties_file)
        
        logger.info(f"Processing {len(properties)} properties in batches...")
        all_results = []
        
        # Process properties in batches
        for i in range(0, len(properties), BATCH_SIZE):
            batch = properties.iloc[i:i + BATCH_SIZE]
            logger.info(f"Processing batch {i//BATCH_SIZE + 1}...")
            
            batch_results = self.process_property_batch(batch)
            all_results.append(batch_results)
        
        # Combine results
        results_df = pd.concat(all_results, ignore_index=True)
        logger.info(f"Completed processing {len(results_df)} properties")
        
        return results_df
    
    def calculate_trends(self, stats_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate NDVI trends for each property.
        
        Args:
            stats_df: DataFrame of NDVI statistics
            
        Returns:
            DataFrame with added trend calculations
        """
        logger.info("Calculating NDVI trends...")
        
        def calculate_property_trend(row):
            years = np.array(ANALYSIS_YEARS)
            ndvi_values = np.array([row[f'ndvi_{year}_mean'] for year in years])
            
            # Calculate linear trend
            slope, intercept = np.polyfit(years, ndvi_values, 1)
            
            return pd.Series({
                'ndvi_trend_slope': slope,
                'ndvi_trend_intercept': intercept,
                'ndvi_trend_r2': np.corrcoef(years, ndvi_values)[0,1]**2
            })
        
        # Calculate trends for each property
        trend_df = stats_df.apply(calculate_property_trend, axis=1)
        results_df = pd.concat([stats_df, trend_df], axis=1)
        
        logger.info("Trend calculation complete")
        return results_df 