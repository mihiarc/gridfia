#!/usr/bin/env python3

import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

class PropertyNDVIAnalyzer:
    """Integrates property boundaries with NDVI data for temporal analysis."""
    
    def __init__(self, data_dir="data"):
        """Initialize the analyzer with data paths and configuration."""
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir = self.data_dir / "raw"
        
        # NDVI file paths (from previous analysis)
        self.ndvi_files = {
            '2018': self.raw_dir / "ndvi_Heirs-Vance_County" / "ndvi_NAIP_Vance_County_2018-p2.tif",
            '2020': self.raw_dir / "ndvi_Heirs-Vance_County" / "ndvi_NAIP_Vance_County_2020-p2.tif",
            '2022': self.raw_dir / "ndvi_Heirs-Vance_County" / "ndvi_NAIP_Vance_County_2022-p2.tif"
        }
        
        # Verify NDVI files exist
        for year, path in self.ndvi_files.items():
            if not path.exists():
                raise FileNotFoundError(f"NDVI file for {year} not found: {path}")
        
        # Load and store NDVI bounds for filtering
        with rasterio.open(self.ndvi_files['2018']) as src:
            self.ndvi_bounds = src.bounds
            self.ndvi_crs = src.crs
    
    def filter_properties_by_county(self, properties_gdf, county_name='Vance'):
        """Filter properties to only those in specified county."""
        return properties_gdf[properties_gdf['county_nam'] == f"{county_name} County"].copy()
    
    def filter_properties_by_bounds(self, properties_gdf):
        """Filter properties to only those that intersect with NDVI bounds."""
        # Ensure properties are in same CRS as NDVI
        original_crs = properties_gdf.crs
        if properties_gdf.crs != self.ndvi_crs:
            print(f"\nReprojecting from {properties_gdf.crs} to {self.ndvi_crs}")
            properties_gdf = properties_gdf.to_crs(self.ndvi_crs)
        
        # Create bounds polygon
        from shapely.geometry import box
        bounds_poly = box(*self.ndvi_bounds)
        
        # Log bounds information
        print(f"\nNDVI bounds:")
        print(f"  Left: {self.ndvi_bounds.left:.6f}")
        print(f"  Right: {self.ndvi_bounds.right:.6f}")
        print(f"  Bottom: {self.ndvi_bounds.bottom:.6f}")
        print(f"  Top: {self.ndvi_bounds.top:.6f}")
        
        # Get property bounds for comparison
        total_bounds = properties_gdf.total_bounds
        print(f"\nProperty bounds (in NDVI CRS):")
        print(f"  Left: {total_bounds[0]:.6f}")
        print(f"  Right: {total_bounds[2]:.6f}")
        print(f"  Bottom: {total_bounds[1]:.6f}")
        print(f"  Top: {total_bounds[3]:.6f}")
        
        # Filter properties that intersect bounds
        print(f"\nFiltering {len(properties_gdf)} properties...")
        mask = properties_gdf.geometry.intersects(bounds_poly)
        filtered = properties_gdf[mask].copy()
        print(f"Found {len(filtered)} properties intersecting NDVI bounds")
        
        # Analyze non-intersecting properties
        non_intersecting = properties_gdf[~mask]
        if len(non_intersecting) > 0:
            print(f"\nAnalyzing {len(non_intersecting)} non-intersecting properties:")
            non_int_bounds = non_intersecting.total_bounds
            print(f"Non-intersecting properties bounds:")
            print(f"  Left: {non_int_bounds[0]:.6f}")
            print(f"  Right: {non_int_bounds[2]:.6f}")
            print(f"  Bottom: {non_int_bounds[1]:.6f}")
            print(f"  Top: {non_int_bounds[3]:.6f}")
        
        return filtered
    
    def load_property_data(self, buffer_distance_km=5.0):
        """Load property data and neighbor relationships."""
        print("\nLoading property data...")
        
        # Load heirs properties
        heirs_df = gpd.read_parquet(self.raw_dir / "nc-hp_v2.parquet")
        print(f"Loaded {len(heirs_df)} total heirs properties")
        
        # Filter heirs properties to Vance County
        heirs_df = self.filter_properties_by_county(heirs_df)
        print(f"Found {len(heirs_df)} heirs properties in Vance County")
        
        # Further filter by NDVI bounds
        heirs_df = self.filter_properties_by_bounds(heirs_df)
        print(f"Found {len(heirs_df)} heirs properties within NDVI coverage")
        
        if len(heirs_df) == 0:
            raise ValueError("No heirs properties found in Vance County within NDVI coverage")
        
        # Load neighbor relationships
        neighbor_file = self.processed_dir / f"neighbors_{buffer_distance_km}km.parquet"
        if not neighbor_file.exists():
            raise FileNotFoundError(f"Neighbor file not found: {neighbor_file}")
        
        neighbors_df = pd.read_parquet(neighbor_file)
        print(f"Loaded {len(neighbors_df)} total neighbor relationships")
        
        # Filter neighbor relationships to only those involving Vance County heirs properties
        neighbors_df = neighbors_df[neighbors_df['heirs_parcel_id'].isin(heirs_df.index)]
        print(f"Found {len(neighbors_df)} neighbor relationships for Vance County heirs properties")
        
        # Get unique neighbor IDs
        neighbor_ids = neighbors_df['neighbor_parcel_id'].unique()
        print(f"Found {len(neighbor_ids)} unique neighbor parcels")
        
        # Load only the necessary neighbor parcels from NC parcels dataset
        print("\nLoading neighbor parcels...")
        parcels_df = gpd.read_parquet(
            self.raw_dir / "nc-parcels.parquet",
            columns=['geometry', 'PARCELAPN', 'FIPS']  # Load only necessary columns
        )
        print(f"Loaded {len(parcels_df)} total NC parcels")
        
        # Filter to only neighbor parcels
        parcels_df = parcels_df.loc[parcels_df.index.isin(neighbor_ids)]
        print(f"Filtered to {len(parcels_df)} neighbor parcels")
        
        # Further filter by NDVI bounds
        parcels_df = self.filter_properties_by_bounds(parcels_df)
        print(f"Found {len(parcels_df)} neighbor parcels within NDVI coverage")
        
        return neighbors_df, heirs_df, parcels_df
    
    def extract_ndvi_statistics(self, geometry, ndvi_data, nodata=None):
        """Extract NDVI statistics for a single geometry."""
        try:
            # Mask NDVI data with geometry
            masked_data, _ = mask(ndvi_data, [geometry], crop=True, nodata=nodata)
            
            # Filter out nodata values
            if nodata is not None:
                valid_data = masked_data[masked_data != nodata]
            else:
                valid_data = masked_data
            
            if len(valid_data) == 0:
                return None
            
            # Calculate statistics
            stats = {
                'mean': float(np.mean(valid_data)),
                'median': float(np.median(valid_data)),
                'std': float(np.std(valid_data)),
                'min': float(np.min(valid_data)),
                'max': float(np.max(valid_data)),
                'count': int(len(valid_data)),
                'percentiles': {
                    '25': float(np.percentile(valid_data, 25)),
                    '75': float(np.percentile(valid_data, 75))
                }
            }
            return stats
            
        except Exception as e:
            print(f"Error extracting NDVI statistics: {str(e)}")
            return None
    
    def process_property(self, property_row, year, ndvi_src):
        """Process a single property for a given year."""
        try:
            stats = self.extract_ndvi_statistics(
                property_row.geometry,
                ndvi_src,
                nodata=ndvi_src.nodata
            )
            
            if stats is None:
                return None
            
            return {
                'property_id': property_row.name,
                'year': year,
                'ndvi_stats': stats
            }
            
        except Exception as e:
            print(f"Error processing property {property_row.name}: {str(e)}")
            return None
    
    def calculate_temporal_trends(self, property_stats):
        """Calculate temporal trends for a property."""
        years = sorted(property_stats.keys())
        if len(years) < 2:
            return None
        
        try:
            # Calculate year-over-year changes
            changes = {}
            for i in range(len(years)-1):
                year1, year2 = years[i], years[i+1]
                change = (
                    property_stats[year2]['ndvi_stats']['mean'] -
                    property_stats[year1]['ndvi_stats']['mean']
                )
                changes[f'change_{year1}_{year2}'] = float(change)
            
            # Calculate overall trend (simple linear regression)
            x = np.array([int(year) for year in years])
            y = np.array([property_stats[year]['ndvi_stats']['mean'] for year in years])
            
            # Check for valid data
            if len(x) < 2 or len(y) < 2 or np.any(np.isnan(y)) or np.any(np.isinf(y)):
                return {
                    'changes': changes,
                    'slope': None,
                    'intercept': None,
                    'error': 'Invalid data for trend calculation'
                }
            
            try:
                slope, intercept = np.polyfit(x, y, 1)
                trends = {
                    'slope': float(slope),
                    'intercept': float(intercept),
                    'changes': changes
                }
            except Exception as e:
                trends = {
                    'changes': changes,
                    'slope': None,
                    'intercept': None,
                    'error': f'Error calculating trend: {str(e)}'
                }
            
            return trends
            
        except Exception as e:
            print(f"Error calculating trends: {str(e)}")
            return None
    
    def process_properties(self, properties_gdf, batch_size=10):
        """Process all properties and calculate NDVI statistics."""
        results = {}
        
        print("\nProcessing properties...")
        for year, ndvi_file in self.ndvi_files.items():
            print(f"\nProcessing year {year}")
            
            # Open raster with memory-mapped reading
            with rasterio.Env(GDAL_CACHEMAX=512):  # Set GDAL cache to 512MB
                with rasterio.open(ndvi_file) as ndvi_src:
                    # Process properties in batches
                    for i in range(0, len(properties_gdf), batch_size):
                        batch = properties_gdf.iloc[i:i+batch_size]
                        print(f"\nProcessing batch {i//batch_size + 1} of {(len(properties_gdf)-1)//batch_size + 1}")
                        
                        # Process each property in the batch
                        for idx, row in tqdm(batch.iterrows(), desc=f"Processing {year}", total=len(batch)):
                            result = self.process_property(row, year, ndvi_src)
                            if result is not None:
                                prop_id = result['property_id']
                                if prop_id not in results:
                                    results[prop_id] = {}
                                results[prop_id][year] = result
        
        # Calculate temporal trends
        print("\nCalculating temporal trends...")
        for prop_id in tqdm(results.keys(), desc="Calculating trends"):
            trends = self.calculate_temporal_trends(results[prop_id])
            if trends is not None:
                results[prop_id]['temporal_trends'] = trends
        
        return results

def main():
    """Main analysis function."""
    analyzer = PropertyNDVIAnalyzer()
    
    try:
        # Load property data
        neighbors_df, heirs_df, parcels_df = analyzer.load_property_data()
        
        # Process heirs properties
        print("\nProcessing heirs properties...")
        heirs_results = analyzer.process_properties(heirs_df)
        
        # Get unique neighbor properties
        neighbor_ids = neighbors_df['neighbor_parcel_id'].unique()
        neighbor_properties = parcels_df.loc[neighbor_ids]
        
        # Process neighbor properties
        print("\nProcessing neighbor properties...")
        neighbor_results = analyzer.process_properties(neighbor_properties)
        
        # Save results
        results = {
            'heirs': heirs_results,
            'neighbors': neighbor_results,
            'metadata': {
                'heirs_count': len(heirs_results),
                'neighbors_count': len(neighbor_results),
                'processing_date': pd.Timestamp.now().isoformat()
            }
        }
        
        output_file = analyzer.processed_dir / "property_ndvi_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nAnalysis complete. Results saved to: {output_file}")
        print(f"Processed {len(heirs_results)} heirs properties and {len(neighbor_results)} neighbor properties")
        
    except Exception as e:
        print(f"Error in main analysis: {str(e)}")
        raise

if __name__ == '__main__':
    main() 