#!/usr/bin/env python3

import pandas as pd
import geopandas as gpd
from pathlib import Path
import numpy as np
import rasterio
from rasterio.mask import mask
import warnings
from typing import Dict, List, Optional, Tuple
from shapely.wkt import loads
import random

class NDVIProcessor:
    """Processes NDVI data for properties."""
    
    def __init__(self, ndvi_dir: str = "data/raw/ndvi"):
        """Initialize the NDVI processor.
        
        Args:
            ndvi_dir: Directory containing NDVI raster files
        """
        self.ndvi_dir = Path(ndvi_dir)
        self.years = ['2018', '2020', '2022']
        self.ndvi_files = {}
        self.target_crs = "EPSG:2264"  # NC State Plane
        self._load_ndvi_files()
        
    def _load_ndvi_files(self):
        """Load NDVI files for each year."""
        print("\nLoading NDVI files...")
        for year in self.years:
            # Get both parts for each year
            files = sorted(self.ndvi_dir.glob(f"*_{year}*.tif"))
            if files:
                self.ndvi_files[year] = files
                print(f"Found {len(files)} NDVI files for {year}:")
                for f in files:
                    print(f"  - {f.name}")
            else:
                warnings.warn(f"No NDVI file found for {year}")
    
    def load_processed_data(self):
        """Load preprocessed property data.
        
        Returns:
            Tuple of (heirs_properties, all_properties) as GeoDataFrames
        """
        processed_dir = Path("output/processed")
        heirs_path = processed_dir / "heirs_processed.parquet"
        parcels_path = processed_dir / "parcels_processed.parquet"
        
        print("\nLoading processed property data...")
        
        try:
            # Load parquet files
            heirs_df = pd.read_parquet(heirs_path)
            parcels_df = pd.read_parquet(parcels_path)
            
            # Convert WKT geometry back to GeoDataFrame
            heirs_gdf = gpd.GeoDataFrame(
                heirs_df,
                geometry=heirs_df['geometry_wkt'].apply(loads),
                crs=self.target_crs
            )
            
            parcels_gdf = gpd.GeoDataFrame(
                parcels_df,
                geometry=parcels_df['geometry_wkt'].apply(loads),
                crs=self.target_crs
            )
            
            print(f"Loaded properties:")
            print(f"- Heirs properties: {len(heirs_gdf)}")
            print(f"- All parcels: {len(parcels_gdf)}")
            
            return heirs_gdf, parcels_gdf
            
        except Exception as e:
            print(f"Error loading processed data: {str(e)}")
            raise
    
    def extract_ndvi(self, geometry, year: int) -> Tuple[Optional[float], int]:
        """Extract NDVI values for a property in a given year.
        
        Args:
            geometry: Shapely geometry object
            year: Year to extract NDVI for
            
        Returns:
            Tuple of (mean NDVI value or None, number of valid pixels)
        """
        # Get list of NDVI files for this year
        ndvi_files = self.ndvi_files.get(str(year), [])
        if not ndvi_files:
            print(f"No NDVI files found for year {year}")
            return None, 0
            
        # Initialize pixel counts and sums
        total_pixels = 0
        ndvi_sum = 0
        
        # Process each NDVI file
        for ndvi_file in ndvi_files:
            try:
                # Read NDVI data
                with rasterio.open(ndvi_file) as src:
                    # Create a GeoDataFrame with the geometry in the same CRS as the raster
                    property_gdf = gpd.GeoDataFrame(geometry=[geometry], crs=self.target_crs)
                    property_gdf = property_gdf.to_crs(src.crs)
                    
                    # Get pixels within property boundary
                    out_image, out_transform = mask(src, property_gdf.geometry, crop=True)
                    
                    # Get valid pixels (not masked)
                    valid_pixels = ~out_image[0].mask if hasattr(out_image[0], 'mask') else np.ones_like(out_image[0], dtype=bool)
                    valid_values = out_image[0][valid_pixels]
                    
                    if len(valid_values) > 0:
                        total_pixels += len(valid_values)
                        ndvi_sum += np.sum(valid_values)
                        print(f"Found {len(valid_values)} valid pixels in {ndvi_file}")
                        
            except Exception as e:
                print(f"Error processing {ndvi_file}: {str(e)}")
                continue
                
        if total_pixels > 0:
            ndvi_value = ndvi_sum / total_pixels
            print(f"Calculated NDVI value for year {year}: {ndvi_value:.4f}")
            return ndvi_value, total_pixels
        else:
            print(f"No valid pixels found for year {year}")
            return None, 0
    
    def process_properties(self, property_ids: List[str]) -> pd.DataFrame:
        """Process a list of properties to extract NDVI values."""
        # Load property data
        heirs_gdf, parcels_gdf = self.load_processed_data()
        
        # Combine properties and add type flag
        heirs_gdf['property_type'] = 'heir'
        parcels_gdf['property_type'] = 'neighbor'
        all_properties = pd.concat([heirs_gdf, parcels_gdf])
        
        # Validate property IDs
        invalid_ids = [pid for pid in property_ids if pid not in all_properties['property_id'].values]
        if invalid_ids:
            print(f"Warning: {len(invalid_ids)} invalid property IDs found")
            print(f"First few invalid IDs: {invalid_ids[:5]}")
            property_ids = [pid for pid in property_ids if pid not in invalid_ids]
        
        if not property_ids:
            print("No valid property IDs to process")
            return pd.DataFrame()
        
        results = []
        total = len(property_ids)

        print(f"\nProcessing {total} properties\n")

        for i, prop_id in enumerate(property_ids, 1):
            print(f"Processing property {i}/{total}")
            
            try:
                # Get property data
                prop_data = all_properties[all_properties['property_id'] == prop_id].iloc[0]
                prop_geom = prop_data['geometry']
                prop_type = prop_data['property_type']
                
                # Process NDVI for each year
                ndvi_values = {}
                pixel_counts = {}
                coverage_ratios = {}
                
                for year in self.years:
                    try:
                        # Extract NDVI
                        ndvi, pixels = self.extract_ndvi(prop_geom, int(year))
                        ndvi_values[f'ndvi_{year}'] = ndvi
                        pixel_counts[f'pixels_{year}'] = pixels
                        
                        # Calculate coverage ratio
                        if pixels and pixels > 0:
                            # Estimate pixel area (assuming 1m resolution)
                            pixel_area = pixels * 1.0  # 1 sq meter per pixel
                            coverage_ratio = pixel_area / prop_data['area']
                            coverage_ratios[f'coverage_{year}'] = coverage_ratio
                        else:
                            coverage_ratios[f'coverage_{year}'] = 0.0
                            
                    except Exception as e:
                        print(f"Error processing property {prop_id} for year {year}: {str(e)}")
                        ndvi_values[f'ndvi_{year}'] = None
                        pixel_counts[f'pixels_{year}'] = 0
                        coverage_ratios[f'coverage_{year}'] = 0.0

                # Generate NDVI record ID
                ndvi_id = f"NDVI_{prop_id}"
                
                results.append({
                    'ndvi_id': ndvi_id,
                    'property_id': prop_id,
                    'property_type': prop_type,
                    **ndvi_values,
                    **pixel_counts,
                    **coverage_ratios
                })
                
            except Exception as e:
                print(f"Error processing property {prop_id}: {str(e)}")
                continue

        return pd.DataFrame(results)

def main():
    """Run NDVI processing as a standalone script."""
    try:
        # Initialize processor
        processor = NDVIProcessor()
        
        # Load matches file
        matches_file = Path("output/matches/vance_matches.csv")
        if not matches_file.exists():
            print("Please run property_matcher.py first")
            return
            
        matches_df = pd.read_csv(matches_file)
        print(f"\nLoaded {len(matches_df)} property matches")
        
        # Get unique property IDs
        property_ids = pd.concat([
            matches_df['heir_id'],
            matches_df['neighbor_id']
        ]).unique()
        print(f"Found {len(property_ids)} unique properties to process")
        
        # Process properties in batches
        batch_size = 100
        results = []
        
        for i in range(0, len(property_ids), batch_size):
            batch_ids = property_ids[i:i + batch_size]
            print(f"\nProcessing batch {i//batch_size + 1} of {(len(property_ids)-1)//batch_size + 1}")
            
            print(f"\nProcessing {len(batch_ids)} properties")
            batch_results = processor.process_properties(batch_ids)
            
            if batch_results is not None and len(batch_results) > 0:
                results.append(batch_results)
        
        # Combine and validate results
        if results:
            results_df = pd.concat(results, ignore_index=True)
            
            # Validate results
            print("\nValidating results...")
            total_properties = len(property_ids)
            processed_properties = len(results_df)
            coverage = (processed_properties / total_properties) * 100
            print(f"Processed {processed_properties} of {total_properties} properties ({coverage:.1f}%)")
            
            # Check NDVI coverage
            for year in processor.years:
                valid_ndvi = results_df[f'ndvi_{year}'].notna()
                coverage = valid_ndvi.mean() * 100
                print(f"Year {year} NDVI coverage: {coverage:.1f}%")
            
            # Split results by property type
            heirs_df = results_df[results_df['property_type'] == 'heir']
            neighbors_df = results_df[results_df['property_type'] == 'neighbor']
            
            # Save results
            output_dir = Path("output/ndvi")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            heirs_df.to_csv(output_dir / "vance_heirs_ndvi.csv", index=False)
            print(f"Saved NDVI values for {len(heirs_df)} heir properties")
            
            neighbors_df.to_csv(output_dir / "vance_neighbors_ndvi.csv", index=False)
            print(f"Saved NDVI values for {len(neighbors_df)} neighbor properties")
        else:
            print("No results to save")
        
    except Exception as e:
        print(f"\nError during NDVI processing: {str(e)}")
        raise

if __name__ == "__main__":
    main() 