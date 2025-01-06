#!/usr/bin/env python3

"""Data preparation module for heirs property analysis.

This module handles the loading and preparation of geospatial data for heirs property analysis.
It provides functionality to:
- Load heirs property and parcel data from GeoJSON files
- Convert coordinate reference systems to NC State Plane (EPSG:2264)
- Validate geometry and area data
- Save processed data in parquet format with WKT geometries

Example:
    >>> preparator = DataPreparator()
    >>> heirs_gdf, parcels_gdf = preparator.load_and_prepare_data()
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path
import numpy as np
import os

class DataPreparator:
    """Handles data loading and preparation for heirs property analysis.
    
    This class provides methods to load and prepare geospatial data for analysis.
    It handles coordinate reference system conversion and basic data validation.
    
    Attributes:
        data_dir (Path): Base directory for data files
        target_crs (str): Target coordinate reference system (default: EPSG:2264 NC State Plane)
    
    Example:
        >>> prep = DataPreparator(data_dir="data")
        >>> heirs_gdf, parcels_gdf = prep.load_and_prepare_data()
    """
    
    def __init__(self, data_dir="data"):
        """Initialize the data preparator.
        
        Args:
            data_dir (str): Base directory for data files. Defaults to "data".
                          Should contain a raw/gis subdirectory with input files.
        """
        self.data_dir = Path(data_dir)
        self.target_crs = "EPSG:2264"  # NC State Plane
        
    def load_and_prepare_data(self):
        """Load and prepare property data for analysis.
        
        Loads heirs property and parcel data from parquet files and prepares them
        for analysis by converting to the target CRS.
        
        Returns:
            tuple: A pair of GeoDataFrames (heirs_gdf, parcels_gdf) containing the
                  prepared property data.
        
        Raises:
            FileNotFoundError: If input files are not found in the expected location.
            ValueError: If the input data is invalid or missing required columns.
        """
        print("Loading property data...")
        
        # Define file paths
        heirs_path = os.path.join(self.data_dir, "raw", "gis", "nc-hp_v2.parquet")
        parcels_path = os.path.join(self.data_dir, "raw", "gis", "nc-parcels.parquet")
        
        # Load data
        print(f"\nLoading heirs properties from: {heirs_path}")
        heirs_gdf = gpd.read_parquet(heirs_path)
        print(f"Loaded {len(heirs_gdf)} heirs properties")
        
        print(f"\nLoading parcels from: {parcels_path}")
        parcels_gdf = gpd.read_parquet(parcels_path)
        print(f"Loaded {len(parcels_gdf)} parcels")
        
        # Convert to target CRS if needed
        if heirs_gdf.crs != self.target_crs:
            print(f"\nConverting heirs properties to {self.target_crs}")
            heirs_gdf = heirs_gdf.to_crs(self.target_crs)
        
        if parcels_gdf.crs != self.target_crs:
            print(f"\nConverting parcels to {self.target_crs}")
            parcels_gdf = parcels_gdf.to_crs(self.target_crs)
        
        # Generate consistent property IDs
        print("\nGenerating property IDs...")
        
        # For heirs properties
        heirs_gdf['base_id'] = heirs_gdf.apply(
            lambda row: f"H_{row.county_nam}_{row.PARCELAPN}" if pd.notna(row.PARCELAPN)
            else f"H_{row.county_nam}_{row.name}",
            axis=1
        )
        
        # Handle duplicates in heirs properties
        heirs_gdf['dup_count'] = heirs_gdf.groupby('base_id').cumcount()
        heirs_gdf['property_id'] = heirs_gdf.apply(
            lambda row: f"{row.base_id}_{row.dup_count}" if row.dup_count > 0
            else row.base_id,
            axis=1
        )
        heirs_gdf = heirs_gdf.drop(['base_id', 'dup_count'], axis=1)
        
        # For parcels
        parcels_gdf['base_id'] = parcels_gdf.apply(
            lambda row: f"P_{row.FIPS}_{row.PARCELAPN}" if pd.notna(row.PARCELAPN)
            else f"P_{row.FIPS}_{row.name}",
            axis=1
        )
        
        # Handle duplicates in parcels
        parcels_gdf['dup_count'] = parcels_gdf.groupby('base_id').cumcount()
        parcels_gdf['property_id'] = parcels_gdf.apply(
            lambda row: f"{row.base_id}_{row.dup_count}" if row.dup_count > 0
            else row.base_id,
            axis=1
        )
        parcels_gdf = parcels_gdf.drop(['base_id', 'dup_count'], axis=1)
        
        # Calculate areas
        print("\nCalculating areas...")
        heirs_gdf['area'] = heirs_gdf.geometry.area
        parcels_gdf['area'] = parcels_gdf.geometry.area
        
        # Add property types
        heirs_gdf['property_type'] = 'heir'
        parcels_gdf['property_type'] = 'neighbor'
        
        # Add county name to parcels
        parcels_gdf['county_nam'] = parcels_gdf['FIPS'].apply(lambda x: f"COUNTY_{x}")
        
        # Validate data
        print("\nValidating data...")
        self._validate_data(heirs_gdf, "heirs")
        self._validate_data(parcels_gdf, "parcels")
        
        return heirs_gdf, parcels_gdf

    def _validate_data(self, gdf: gpd.GeoDataFrame, dataset_name: str):
        """Validate GeoDataFrame.
        
        Args:
            gdf: GeoDataFrame to validate
            dataset_name: Name of dataset for logging
        
        Raises:
            ValueError: If validation fails
        """
        # Check for required columns
        required_cols = ['property_id', 'geometry', 'area', 'property_type', 'county_nam']
        missing_cols = [col for col in required_cols if col not in gdf.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in {dataset_name}: {missing_cols}")
        
        # Check for duplicate IDs
        duplicates = gdf['property_id'].duplicated()
        if duplicates.any():
            raise ValueError(
                f"Found {duplicates.sum()} duplicate property IDs in {dataset_name}"
            )
        
        # Validate geometry
        invalid_geom = ~gdf.geometry.is_valid
        if invalid_geom.any():
            print(f"Warning: Found {invalid_geom.sum()} invalid geometries in {dataset_name}")
        
        # Validate areas
        zero_area = gdf['area'] <= 0
        if zero_area.any():
            print(f"Warning: Found {zero_area.sum()} zero/negative areas in {dataset_name}")
        
        print(f"Validated {len(gdf)} {dataset_name} records")

def main():
    """Run the data preparation pipeline as a standalone script.
    
    This function:
    1. Loads heirs property and parcel data
    2. Converts geometries to WKT format
    3. Saves processed data to parquet files in the output/processed directory
    
    The output files will be:
    - output/processed/heirs_processed.parquet
    - output/processed/parcels_processed.parquet
    
    Raises:
        FileNotFoundError: If input files are not found
        Exception: If any error occurs during processing
    """
    try:
        preparator = DataPreparator()
        heirs_gdf, parcels_gdf = preparator.load_and_prepare_data()
        
        # Save processed data
        output_dir = Path("output/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving to directory: {output_dir.absolute()}")
        
        # Convert geometries to WKT strings
        print("Converting geometries to WKT format...")
        heirs_df = heirs_gdf.copy()
        parcels_df = parcels_gdf.copy()
        
        heirs_df['geometry_wkt'] = heirs_df['geometry'].apply(lambda x: x.wkt if x else None)
        parcels_df['geometry_wkt'] = parcels_df['geometry'].apply(lambda x: x.wkt if x else None)
        
        # Drop geometry column before saving
        heirs_df = heirs_df.drop(columns=['geometry'])
        parcels_df = parcels_df.drop(columns=['geometry'])
        
        # Save as parquet with error handling
        print("Saving heirs properties...")
        heirs_output = output_dir / "heirs_processed.parquet"
        heirs_df.to_parquet(heirs_output, index=False)
        print(f"Saved heirs properties to: {heirs_output}")
        
        print("Saving all parcels...")
        parcels_output = output_dir / "parcels_processed.parquet"
        parcels_df.to_parquet(parcels_output, index=False)
        print(f"Saved parcels to: {parcels_output}")
        
        # Verify files exist
        if heirs_output.exists() and parcels_output.exists():
            print("\nSuccessfully saved all files:")
            print(f"- Heirs properties ({heirs_output.stat().st_size / 1024 / 1024:.1f} MB)")
            print(f"- All parcels ({parcels_output.stat().st_size / 1024 / 1024:.1f} MB)")
        else:
            print("\nWarning: Some files may not have been saved correctly")
            
    except Exception as e:
        print(f"\nError during data preparation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 