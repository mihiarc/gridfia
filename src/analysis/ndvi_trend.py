import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from typing import Dict, List, Tuple, Optional
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NDVITrendAnalyzer:
    """Analyzes temporal NDVI trends for properties using mosaicked data."""
    
    def __init__(self, mosaic_paths: Dict[str, str], output_dir: str, n_workers: Optional[int] = None):
        """
        Initialize the NDVI trend analyzer.
        
        Args:
            mosaic_paths: Dictionary mapping years to mosaic file paths
            output_dir: Directory to save results
            n_workers: Number of worker processes (default: CPU count - 1)
        """
        self.mosaic_paths = mosaic_paths
        self.output_dir = output_dir
        self.n_workers = n_workers or max(1, multiprocessing.cpu_count() - 1)
        os.makedirs(output_dir, exist_ok=True)
        
    def _extract_ndvi_stats(self, geometry: Dict, year: str, property_id: str) -> Dict:
        """
        Extract NDVI statistics for a single geometry and year.
        
        Args:
            geometry: GeoJSON-like geometry dictionary
            year: Year to process
            property_id: ID of the property being processed
            
        Returns:
            Dictionary of NDVI statistics with property ID
        """
        mosaic_path = self.mosaic_paths[year]
        
        with rasterio.open(mosaic_path) as src:
            try:
                # Convert geometry to raster CRS if needed
                if isinstance(geometry, dict):
                    # If geometry is already a GeoJSON dict, use it directly
                    geom = geometry
                else:
                    # If geometry is a shapely geometry, convert to GeoJSON
                    geom = geometry.__geo_interface__
                
                # Mask the raster to the geometry
                masked_data, masked_transform = mask(src, [geom], crop=True)
                
                # Extract valid pixels
                valid_pixels = masked_data[0][masked_data[0] != src.nodata]
                
                if len(valid_pixels) > 0:
                    return {
                        'property_id': property_id,  # Include property ID from the start
                        'year': year,
                        'mean_ndvi': float(np.mean(valid_pixels)),
                        'median_ndvi': float(np.median(valid_pixels)),
                        'std_ndvi': float(np.std(valid_pixels)),
                        'min_ndvi': float(np.min(valid_pixels)),
                        'max_ndvi': float(np.max(valid_pixels)),
                        'pixel_count': len(valid_pixels),
                        'has_data': True
                    }
            except Exception as e:
                logger.warning(f"Error processing geometry for property {property_id}, year {year}: {e}")
        
        return {
            'property_id': property_id,  # Include property ID even in error case
            'year': year,
            'mean_ndvi': None,
            'median_ndvi': None,
            'std_ndvi': None,
            'min_ndvi': None,
            'max_ndvi': None,
            'pixel_count': 0,
            'has_data': False
        }
    
    def _process_single_property(self, property_data: Tuple[str, Dict]) -> Dict:
        """
        Process NDVI trends for a single property.
        
        Args:
            property_data: Tuple of (property_id, geometry)
            
        Returns:
            Dictionary of NDVI trends
        """
        property_id, geometry = property_data
        
        # Extract NDVI for each year
        yearly_stats = []
        for year in sorted(self.mosaic_paths.keys()):
            stats = self._extract_ndvi_stats(geometry, year, property_id)  # Pass property_id
            yearly_stats.append(stats)
        
        # Calculate trends
        df = pd.DataFrame(yearly_stats)
        valid_years = df[df['has_data']].sort_values('year')
        
        if len(valid_years) >= 2:
            # Calculate linear trend
            x = pd.to_numeric(valid_years['year'])
            y = valid_years['mean_ndvi']
            slope, intercept = np.polyfit(x, y, 1)
            
            trend_stats = {
                'property_id': property_id,
                'slope': float(slope),
                'intercept': float(intercept),
                'years_covered': len(valid_years),
                'total_pixels': int(valid_years['pixel_count'].sum()),
                'mean_coverage': float(valid_years['pixel_count'].mean()),
                'start_ndvi': float(valid_years.iloc[0]['mean_ndvi']),
                'end_ndvi': float(valid_years.iloc[-1]['mean_ndvi']),
                'total_change': float(valid_years.iloc[-1]['mean_ndvi'] - valid_years.iloc[0]['mean_ndvi']),
                'yearly_stats': yearly_stats
            }
        else:
            trend_stats = {
                'property_id': property_id,
                'slope': None,
                'intercept': None,
                'years_covered': len(valid_years),
                'total_pixels': int(df['pixel_count'].sum()),
                'mean_coverage': float(df['pixel_count'].mean()),
                'start_ndvi': None,
                'end_ndvi': None,
                'total_change': None,
                'yearly_stats': yearly_stats
            }
        
        return trend_stats
    
    def process_properties(self, properties_gdf: gpd.GeoDataFrame, batch_size: int = 100) -> pd.DataFrame:
        """
        Process NDVI trends for multiple properties in parallel.
        
        Args:
            properties_gdf: GeoDataFrame containing properties
            batch_size: Number of properties to process in each batch
            
        Returns:
            DataFrame containing NDVI trends
        """
        all_results = []
        total_properties = len(properties_gdf)
        
        # Process properties in batches
        for start_idx in range(0, total_properties, batch_size):
            end_idx = min(start_idx + batch_size, total_properties)
            batch = properties_gdf.iloc[start_idx:end_idx]
            
            # Prepare property data for parallel processing
            property_data = [
                (str(row.property_id), row.geometry)
                for _, row in batch.iterrows()
            ]
            
            # Process batch in parallel
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                batch_results = list(executor.map(self._process_single_property, property_data))
            
            all_results.extend(batch_results)
            logger.info(f"Processed {end_idx}/{total_properties} properties")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame([
            {k: v for k, v in result.items() if k != 'yearly_stats'}
            for result in all_results
        ])
        
        # Save yearly statistics separately
        yearly_stats_df = pd.DataFrame([
            stat for result in all_results
            for stat in result['yearly_stats']
        ])
        
        # Save results
        results_df.to_parquet(os.path.join(self.output_dir, 'ndvi_trends.parquet'))
        yearly_stats_df.to_parquet(os.path.join(self.output_dir, 'ndvi_yearly_stats.parquet'))
        
        return results_df 