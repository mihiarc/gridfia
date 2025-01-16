#!/usr/bin/env python3

"""Data integration module for heirs property analysis.

This module handles data integration and validation across the analysis pipeline,
ensuring data consistency and proper relationships between datasets.
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

# Define schemas for validation
PROPERTY_SCHEMA = {
    'property_id': 'string',
    'parcel_id': 'string',
    'county_name': 'string',
    'geometry': 'geometry',
    'area': 'float64',
    'property_type': 'category'
}

MATCH_SCHEMA = {
    'match_id': 'string',
    'heir_id': 'string',
    'neighbor_id': 'string',
    'distance': 'float64',
    'area_ratio': 'float64'
}

NDVI_SCHEMA = {
    'ndvi_id': 'string',
    'property_id': 'string',
    'year': 'int32',
    'ndvi_value': 'float64',
    'pixel_count': 'int32',
    'coverage_ratio': 'float64'
}

class DataIntegrator:
    """Handles data integration and validation across analysis pipeline."""
    
    def __init__(self):
        """Initialize the data integrator."""
        self.schemas = {
            'property': PROPERTY_SCHEMA,
            'match': MATCH_SCHEMA,
            'ndvi': NDVI_SCHEMA
        }
        self.output_dir = Path("output/integrated")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def validate_schema(self, df: pd.DataFrame, schema_type: str) -> Tuple[bool, Dict]:
        """Validate dataframe against schema.
        
        Args:
            df: DataFrame to validate
            schema_type: Type of schema to validate against
            
        Returns:
            Tuple of (is_valid, validation_results)
        """
        schema = self.schemas[schema_type]
        results = {'missing_columns': [], 'type_mismatches': []}
        
        # Check required columns
        for col, dtype in schema.items():
            if col not in df.columns:
                results['missing_columns'].append(col)
            elif dtype != 'geometry' and str(df[col].dtype) != dtype:
                results['type_mismatches'].append(
                    f"{col}: expected {dtype}, got {df[col].dtype}"
                )
        
        is_valid = (
            len(results['missing_columns']) == 0 and 
            len(results['type_mismatches']) == 0
        )
        
        return is_valid, results
    
    def validate_relationships(self, properties_df: pd.DataFrame, 
                             matches_df: Optional[pd.DataFrame] = None,
                             ndvi_df: Optional[pd.DataFrame] = None) -> Dict:
        """Validate relationships between datasets.
        
        Args:
            properties_df: Property data
            matches_df: Optional match data
            ndvi_df: Optional NDVI data
            
        Returns:
            Dictionary of validation results
        """
        results = {'invalid_references': [], 'duplicate_keys': []}
        
        # Check for duplicate property IDs
        duplicate_props = properties_df['property_id'].duplicated()
        if duplicate_props.any():
            results['duplicate_keys'].append(
                f"Found {duplicate_props.sum()} duplicate property IDs"
            )
        
        if matches_df is not None:
            # Verify heir and neighbor IDs exist in properties
            prop_ids = set(properties_df['property_id'])
            invalid_heirs = matches_df[~matches_df['heir_id'].isin(prop_ids)]
            invalid_neighbors = matches_df[~matches_df['neighbor_id'].isin(prop_ids)]
            
            if len(invalid_heirs) > 0:
                results['invalid_references'].append(
                    f"Found {len(invalid_heirs)} invalid heir IDs"
                )
            if len(invalid_neighbors) > 0:
                results['invalid_references'].append(
                    f"Found {len(invalid_neighbors)} invalid neighbor IDs"
                )
        
        if ndvi_df is not None:
            # Verify property IDs exist
            invalid_ndvi = ndvi_df[~ndvi_df['property_id'].isin(prop_ids)]
            if len(invalid_ndvi) > 0:
                results['invalid_references'].append(
                    f"Found {len(invalid_ndvi)} invalid property IDs in NDVI data"
                )
        
        return results
    
    def merge_datasets(self, properties_df: pd.DataFrame,
                      matches_df: pd.DataFrame,
                      ndvi_df: pd.DataFrame) -> pd.DataFrame:
        """Merge datasets with validation.
        
        Args:
            properties_df: Property data
            matches_df: Match data
            ndvi_df: NDVI data
            
        Returns:
            Merged DataFrame
        """
        # Validate relationships first
        rel_results = self.validate_relationships(
            properties_df, matches_df, ndvi_df
        )
        if rel_results['invalid_references'] or rel_results['duplicate_keys']:
            raise ValueError(
                f"Data validation failed: {rel_results}"
            )
        
        # Perform merges
        merged_data = (
            matches_df
            .merge(
                properties_df,
                left_on='heir_id',
                right_on='property_id',
                validate='many_to_one',
                suffixes=('', '_heir')
            )
            .merge(
                properties_df,
                left_on='neighbor_id',
                right_on='property_id',
                validate='many_to_one',
                suffixes=('', '_neighbor')
            )
        )
        
        # Add NDVI data for both heir and neighbor properties
        heir_ndvi = ndvi_df.copy()
        heir_ndvi = heir_ndvi.add_suffix('_heir')
        heir_ndvi = heir_ndvi.rename(columns={'property_id_heir': 'heir_id'})
        
        neighbor_ndvi = ndvi_df.copy()
        neighbor_ndvi = neighbor_ndvi.add_suffix('_neighbor')
        neighbor_ndvi = neighbor_ndvi.rename(columns={'property_id_neighbor': 'neighbor_id'})
        
        merged_data = (
            merged_data
            .merge(heir_ndvi, on='heir_id', how='left')
            .merge(neighbor_ndvi, on='neighbor_id', how='left')
        )
        
        return merged_data
    
    def save_integrated_data(self, merged_df: pd.DataFrame, filename: str):
        """Save integrated dataset.
        
        Args:
            merged_df: Merged DataFrame to save
            filename: Output filename
        """
        output_path = self.output_dir / filename
        merged_df.to_parquet(output_path, index=False)
        print(f"Saved integrated data to {output_path}")

def main():
    """Run data integration as standalone script."""
    try:
        integrator = DataIntegrator()
        
        # Load datasets
        print("\nLoading datasets...")
        properties_df = pd.read_parquet("output/processed/heirs_processed.parquet")
        matches_df = pd.read_csv("output/matches/vance_matches.csv")
        ndvi_df = pd.concat([
            pd.read_csv("output/ndvi/vance_heirs_ndvi.csv"),
            pd.read_csv("output/ndvi/vance_neighbors_ndvi.csv")
        ])
        
        # Validate schemas
        print("\nValidating schemas...")
        for df, name, schema in [
            (properties_df, 'Properties', 'property'),
            (matches_df, 'Matches', 'match'),
            (ndvi_df, 'NDVI', 'ndvi')
        ]:
            is_valid, results = integrator.validate_schema(df, schema)
            print(f"\n{name} validation results:")
            print(f"Valid: {is_valid}")
            if not is_valid:
                print("Issues found:")
                for k, v in results.items():
                    if v:
                        print(f"- {k}: {v}")
        
        # Validate relationships
        print("\nValidating relationships...")
        rel_results = integrator.validate_relationships(
            properties_df, matches_df, ndvi_df
        )
        print("Relationship validation results:")
        for k, v in rel_results.items():
            if v:
                print(f"- {k}: {v}")
        
        # Merge datasets
        print("\nMerging datasets...")
        merged_data = integrator.merge_datasets(
            properties_df, matches_df, ndvi_df
        )
        
        # Save results
        print("\nSaving integrated data...")
        integrator.save_integrated_data(
            merged_data, "vance_integrated_data.parquet"
        )
        
    except Exception as e:
        print(f"\nError during data integration: {str(e)}")
        raise

if __name__ == "__main__":
    main() 