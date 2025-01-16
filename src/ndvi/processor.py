#!/usr/bin/env python3

from dataclasses import dataclass
from pathlib import Path
import logging
from typing import List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import pandas as pd
from shapely.geometry import box

logger = logging.getLogger(__name__)

@dataclass
class NDVIConfig:
    """Configuration for NDVI processing."""
    temporal_window: int = 5  # years
    save_intermediates: bool = False
    chunk_size: int = 1000  # properties per chunk

class NDVIProcessor:
    """Processes NDVI data for properties."""
    
    def __init__(self, config: NDVIConfig):
        """Initialize the NDVI processor.
        
        Args:
            config: NDVIConfig object with processing parameters
        """
        self.config = config
        
    def process_property_ndvi(self, 
                            property_data: gpd.GeoDataFrame,
                            ndvi_path: Path,
                            output_dir: Path = Path("data/processed/ndvi")) -> pd.DataFrame:
        """Process NDVI data for a set of properties.
        
        Args:
            property_data: GeoDataFrame of properties
            ndvi_path: Path to NDVI raster file
            output_dir: Directory to save results
            
        Returns:
            DataFrame with NDVI statistics for each property
        """
        results = []
        
        with rasterio.open(ndvi_path) as src:
            # Process properties in chunks to manage memory
            for chunk_start in range(0, len(property_data), self.config.chunk_size):
                chunk_end = min(chunk_start + self.config.chunk_size, len(property_data))
                chunk = property_data.iloc[chunk_start:chunk_end]
                
                chunk_results = self._process_chunk(chunk, src)
                results.extend(chunk_results)
                
                logger.info(f"Processed NDVI for properties {chunk_start} to {chunk_end}")
        
        results_df = pd.DataFrame(results)
        
        # Save results
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        output_path = output_dir / f"ndvi_stats_{Path(ndvi_path).stem}.parquet"
        results_df.to_parquet(output_path)
        
        return results_df
    
    def _process_chunk(self, properties: gpd.GeoDataFrame, 
                      ndvi_src: rasterio.DatasetReader) -> List[dict]:
        """Process NDVI for a chunk of properties.
        
        Args:
            properties: GeoDataFrame chunk of properties
            ndvi_src: Open rasterio dataset for NDVI
            
        Returns:
            List of dictionaries with NDVI statistics
        """
        results = []
        
        for idx, prop in properties.iterrows():
            try:
                # Mask NDVI raster with property geometry
                masked_ndvi, masked_transform = mask(ndvi_src, [prop.geometry], crop=True)
                
                if masked_ndvi.size == 0:
                    logger.warning(f"No NDVI data for property {prop.id}")
                    continue
                
                # Calculate statistics
                stats = {
                    'property_id': prop.id,
                    'mean_ndvi': float(np.nanmean(masked_ndvi)),
                    'median_ndvi': float(np.nanmedian(masked_ndvi)),
                    'std_ndvi': float(np.nanstd(masked_ndvi)),
                    'min_ndvi': float(np.nanmin(masked_ndvi)),
                    'max_ndvi': float(np.nanmax(masked_ndvi)),
                    'pixel_count': int(np.sum(~np.isnan(masked_ndvi)))
                }
                
                results.append(stats)
                
            except Exception as e:
                logger.error(f"Error processing property {prop.id}: {str(e)}")
                continue
                
        return results
    
    def calculate_temporal_trends(self, 
                                ndvi_stats: List[pd.DataFrame],
                                dates: List[str]) -> pd.DataFrame:
        """Calculate temporal NDVI trends.
        
        Args:
            ndvi_stats: List of DataFrames with NDVI statistics for different dates
            dates: List of dates corresponding to NDVI statistics
            
        Returns:
            DataFrame with temporal trends
        """
        if len(ndvi_stats) != len(dates):
            raise ValueError("Number of NDVI statistics must match number of dates")
            
        # Combine statistics across dates
        for stats_df, date in zip(ndvi_stats, dates):
            stats_df['date'] = date
            
        combined_stats = pd.concat(ndvi_stats)
        
        # Calculate trends
        trends = []
        
        for property_id in combined_stats['property_id'].unique():
            property_stats = combined_stats[
                combined_stats['property_id'] == property_id
            ].sort_values('date')
            
            if len(property_stats) < 2:
                continue
                
            # Calculate trend statistics
            trend = {
                'property_id': property_id,
                'ndvi_slope': np.polyfit(range(len(dates)), 
                                       property_stats['mean_ndvi'], 1)[0],
                'ndvi_variance': property_stats['mean_ndvi'].var(),
                'start_ndvi': property_stats.iloc[0]['mean_ndvi'],
                'end_ndvi': property_stats.iloc[-1]['mean_ndvi'],
                'total_change': (property_stats.iloc[-1]['mean_ndvi'] - 
                               property_stats.iloc[0]['mean_ndvi'])
            }
            
            trends.append(trend)
            
        return pd.DataFrame(trends) 