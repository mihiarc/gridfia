#!/usr/bin/env python3

import pandas as pd
import geopandas as gpd
from pathlib import Path
import json
from shapely.geometry import box
import numpy as np

class PropertyAnalyzer:
    """Analyzes heirs property and neighbor relationships in Vance County."""
    
    def __init__(self, data_dir="../data"):
        self.data_dir = Path(data_dir)
        # Vance County bounds from NDVI analysis
        self.vance_bounds = box(
            -78.51124666236885,  # west
            36.16363110639788,   # south
            -78.27717264878582,  # east
            36.24948309810118    # north
        )
    
    def load_heirs_properties(self):
        """Load and filter heirs properties for Vance County."""
        heirs_path = self.data_dir / "raw" / "nc-hp_v2.parquet"
        print(f"Loading heirs properties from {heirs_path}")
        
        # Load the parquet file
        heirs_df = gpd.read_parquet(heirs_path)
        
        # Filter for Vance County
        vance_heirs = heirs_df[heirs_df.geometry.intersects(self.vance_bounds)]
        
        print(f"Found {len(vance_heirs)} heirs properties in Vance County")
        return vance_heirs
    
    def load_neighbors(self):
        """Load and filter neighboring parcels."""
        neighbors_path = self.data_dir / "raw" / "parcels_within_1_mile.parquet"
        print(f"Loading neighboring parcels from {neighbors_path}")
        
        # Load the parquet file
        neighbors_df = gpd.read_parquet(neighbors_path)
        
        # Filter for Vance County
        vance_neighbors = neighbors_df[neighbors_df.geometry.intersects(self.vance_bounds)]
        
        print(f"Found {len(vance_neighbors)} neighboring parcels in Vance County")
        return vance_neighbors
    
    def analyze_spatial_relationships(self, heirs_gdf, neighbors_gdf):
        """Analyze spatial relationships between heirs properties and neighbors."""
        results = {
            'heirs_count': len(heirs_gdf),
            'neighbors_count': len(neighbors_gdf),
            'heirs_area_stats': {
                'mean': float(heirs_gdf.geometry.area.mean()),
                'std': float(heirs_gdf.geometry.area.std()),
                'min': float(heirs_gdf.geometry.area.min()),
                'max': float(heirs_gdf.geometry.area.max())
            },
            'neighbor_area_stats': {
                'mean': float(neighbors_gdf.geometry.area.mean()),
                'std': float(neighbors_gdf.geometry.area.std()),
                'min': float(neighbors_gdf.geometry.area.min()),
                'max': float(neighbors_gdf.geometry.area.max())
            }
        }
        return results

    def validate_geometries(self, gdf):
        """Validate geometry quality of the GeoDataFrame."""
        validation = {
            'total_count': len(gdf),
            'valid_count': sum(gdf.geometry.is_valid),
            'invalid_count': sum(~gdf.geometry.is_valid),
            'null_count': sum(gdf.geometry.isna()),
            'geometry_types': gdf.geometry.type.value_counts().to_dict()
        }
        return validation

def main():
    """Main analysis function."""
    analyzer = PropertyAnalyzer()
    
    # Load and analyze heirs properties
    print("\nAnalyzing heirs properties...")
    heirs_gdf = analyzer.load_heirs_properties()
    heirs_validation = analyzer.validate_geometries(heirs_gdf)
    print("\nHeirs properties validation:")
    print(json.dumps(heirs_validation, indent=2))
    
    # Load and analyze neighboring parcels
    print("\nAnalyzing neighboring parcels...")
    neighbors_gdf = analyzer.load_neighbors()
    neighbors_validation = analyzer.validate_geometries(neighbors_gdf)
    print("\nNeighboring parcels validation:")
    print(json.dumps(neighbors_validation, indent=2))
    
    # Analyze spatial relationships
    print("\nAnalyzing spatial relationships...")
    spatial_analysis = analyzer.analyze_spatial_relationships(heirs_gdf, neighbors_gdf)
    print("\nSpatial analysis results:")
    print(json.dumps(spatial_analysis, indent=2))

if __name__ == '__main__':
    main() 