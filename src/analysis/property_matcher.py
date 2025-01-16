#!/usr/bin/env python3

import pandas as pd
import geopandas as gpd
from pathlib import Path
import numpy as np
from shapely.wkt import loads
import multiprocessing as mp
from typing import List, Tuple
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PropertyMatcher:
    """Matches heirs properties with comparable non-heirs properties."""
    
    def __init__(self, min_size_ratio=0.6, max_size_ratio=1.4, distance_threshold=2000, n_workers=None):
        """Initialize the property matcher.
        
        Args:
            min_size_ratio: Minimum ratio of neighbor area to heir area (default: 0.6)
            max_size_ratio: Maximum ratio of neighbor area to heir area (default: 1.4)
            distance_threshold: Maximum distance in meters (default: 2000)
            n_workers: Number of worker processes (default: None, uses min(8, CPU count - 1))
        """
        self.min_size_ratio = min_size_ratio
        self.max_size_ratio = max_size_ratio
        self.distance_threshold = distance_threshold
        # Use at most 8 workers to avoid overwhelming the system
        self.n_workers = n_workers if n_workers is not None else min(8, max(1, mp.cpu_count() - 1))
        
    def load_processed_data(self):
        """Load preprocessed property data.
        
        Returns:
            Tuple of (heirs_properties, all_properties) as GeoDataFrames
        """
        processed_dir = Path("output/processed")
        heirs_path = processed_dir / "heirs_processed.parquet"
        parcels_path = processed_dir / "parcels_processed.parquet"
        
        logger.info("Loading processed property data...")
        
        # Load parquet files as GeoDataFrames
        heirs_gdf = gpd.read_parquet(heirs_path)
        parcels_gdf = gpd.read_parquet(parcels_path)
        
        logger.info(f"\nLoaded properties:")
        logger.info(f"Heirs properties: {len(heirs_gdf)}")
        logger.info(f"All parcels: {len(parcels_gdf)}")
        
        return heirs_gdf, parcels_gdf
    
    def find_matches(self, heir_property: gpd.GeoDataFrame, all_properties: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Find matching properties for a single heir property.
        
        Args:
            heir_property: Single heir property as GeoDataFrame
            all_properties: All properties to search for matches
            
        Returns:
            GeoDataFrame of matching properties
        """
        if len(heir_property) != 1:
            raise ValueError("heir_property must contain exactly one property")
            
        heir_id = heir_property['property_id'].iloc[0]
        
        # Calculate areas if needed
        heir_area = heir_property.geometry.area.iloc[0]
        if 'area' not in all_properties.columns:
            all_properties['area'] = all_properties.geometry.area
        
        logger.info(f"\nFinding matches for heir property {heir_id}")
        logger.info(f"  Heir area: {heir_area:.1f} sq meters")
        
        # Define size range
        min_size = heir_area * self.min_size_ratio
        max_size = heir_area * self.max_size_ratio
        logger.info(f"  Size range: {min_size:.1f} to {max_size:.1f} sq meters")
        
        # Filter by size first
        size_matches = all_properties[
            (all_properties['area'] >= min_size) &
            (all_properties['area'] <= max_size) &
            (all_properties['property_id'] != heir_id)  # Exclude the heir property itself
        ].copy()
        
        logger.info(f"  Found {len(size_matches)} properties within size range")
        
        if len(size_matches) == 0:
            return gpd.GeoDataFrame(geometry=[])
        
        # Calculate distances
        size_matches['distance'] = size_matches.geometry.distance(heir_property.geometry.iloc[0])
        
        # Filter by distance
        distance_matches = size_matches[size_matches['distance'] <= self.distance_threshold]
        logger.info(f"  Found {len(distance_matches)} properties within {self.distance_threshold}m")
        
        if len(distance_matches) == 0:
            return gpd.GeoDataFrame(geometry=[])
        
        # Sort by distance
        distance_matches = distance_matches.sort_values('distance')
        
        # Add match metadata
        distance_matches['heir_id'] = heir_id
        distance_matches['area_ratio'] = distance_matches['area'] / heir_area
        distance_matches['match_id'] = distance_matches.apply(
            lambda row: f"{row['heir_id']}_{row['property_id']}", 
            axis=1
        )
        
        return distance_matches
    
    def _process_batch(self, batch_data: Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]) -> List[pd.DataFrame]:
        """Process a batch of heir properties to find matches.
        
        Args:
            batch_data: Tuple of (heirs_batch, all_properties)
            
        Returns:
            List of DataFrames containing match information
        """
        heirs_batch, all_properties = batch_data
        batch_matches = []
        
        for idx, heir in heirs_batch.iterrows():
            # Create single-property GeoDataFrame using pandas native transpose
            heir_df = pd.DataFrame(heir).transpose()
            heir_gdf = gpd.GeoDataFrame(
                heir_df,
                geometry='geometry',
                crs=heirs_batch.crs
            )
            
            # Find matches
            matches = self.find_matches(heir_gdf, all_properties)
            
            if len(matches) > 0:
                # Extract relevant columns
                match_data = matches[[
                    'match_id', 'heir_id', 'property_id', 'distance', 'area', 'area_ratio'
                ]].copy()
                match_data = match_data.rename(columns={'property_id': 'neighbor_id'})
                batch_matches.append(match_data)
                
        return batch_matches

    def find_all_matches(self, heirs_gdf: gpd.GeoDataFrame, all_properties: gpd.GeoDataFrame, 
                        sample_size: int = None, batch_size: int = 10) -> pd.DataFrame:
        """Find matches for multiple heir properties using parallel processing.
        
        Args:
            heirs_gdf: GeoDataFrame of heir properties
            all_properties: GeoDataFrame of all properties
            sample_size: Optional number of heir properties to process
            batch_size: Number of properties to process in each batch (default: 10)
            
        Returns:
            DataFrame with match information
        """
        # Validate input data
        required_cols = ['property_id', 'geometry', 'area']
        for df, name in [(heirs_gdf, 'heirs'), (all_properties, 'all_properties')]:
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in {name}: {missing_cols}")
        
        if sample_size:
            heirs_sample = heirs_gdf.sample(n=min(sample_size, len(heirs_gdf)))
            logger.info(f"\nProcessing {len(heirs_sample)} sample heir properties")
        else:
            heirs_sample = heirs_gdf
            logger.info(f"\nProcessing all {len(heirs_sample)} heir properties")
        
        # Split heirs into batches
        n_batches = max(1, len(heirs_sample) // batch_size)
        heirs_batches = np.array_split(heirs_sample, n_batches)
        logger.info(f"Split properties into {len(heirs_batches)} batches of ~{batch_size} properties each")
        
        # Prepare batch data
        batch_data = [(batch, all_properties) for batch in heirs_batches]
        
        # Process batches in parallel
        logger.info(f"Processing batches using {self.n_workers} workers...")
        with mp.Pool(self.n_workers) as pool:
            batch_results = pool.map(self._process_batch, batch_data)
        
        # Flatten results
        all_matches = [match for batch in batch_results for match in batch]
        
        if not all_matches:
            logger.info("No matches found")
            return pd.DataFrame()
        
        # Combine all matches
        result = pd.concat(all_matches, ignore_index=True)
        logger.info(f"\nFound {len(result)} total matches")
        
        # Validate results
        duplicate_matches = result['match_id'].duplicated()
        if duplicate_matches.any():
            logger.warning(f"Found {duplicate_matches.sum()} duplicate matches")
        
        return result

def main():
    """Run property matching as a standalone script."""
    try:
        # Initialize matcher with parallel processing
        n_workers = min(8, max(1, mp.cpu_count() - 1))  # Use at most 8 workers
        logger.info(f"\nInitializing matcher with {n_workers} workers")
        matcher = PropertyMatcher(n_workers=n_workers)
        
        # Load processed data
        logger.info("\nLoading processed data...")
        heirs_gdf, parcels_gdf = matcher.load_processed_data()
        
        # Filter for Vance County properties
        logger.info("\nFiltering for Vance County properties...")
        vance_mask = heirs_gdf['county_nam'].str.upper().str.contains('VANCE', na=False)
        vance_properties = heirs_gdf[vance_mask].copy()
        logger.info(f"Found {len(vance_properties)} properties in Vance County")
        
        if len(vance_properties) == 0:
            logger.info("No Vance County properties found")
            return
        
        # Find matches for properties
        logger.info("\nFinding matches...")
        matches = matcher.find_all_matches(vance_properties, parcels_gdf)
        
        if len(matches) == 0:
            logger.info("No matches found")
            return
        
        # Save results
        output_dir = Path("output/matches")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "vance_matches.csv"
        matches.to_csv(output_file, index=False)
        
        logger.info(f"\nSaved {len(matches)} matches to {output_file}")
        logger.info("\nMatch statistics:")
        logger.info(matches.describe())
        
        # Print match summary
        logger.info("\nMatching summary:")
        logger.info(f"Average matches per property: {len(matches) / matches['heir_id'].nunique():.1f}")
        logger.info(f"Average distance: {matches['distance'].mean():.1f}m")
        logger.info(f"Average area ratio: {matches['area_ratio'].mean():.2f}")
        
        # Validate coverage
        total_heirs = len(vance_properties)
        matched_heirs = matches['heir_id'].nunique()
        coverage = (matched_heirs / total_heirs) * 100
        logger.info(f"\nCoverage: {coverage:.1f}% of Vance County heir properties have matches")
        
    except Exception as e:
        logger.error(f"\nError during property matching: {str(e)}")
        raise

if __name__ == "__main__":
    main() 