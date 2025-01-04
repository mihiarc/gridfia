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
        # NC State Plane (meters)
        self.target_crs = "EPSG:2264"
        
    def load_heirs_properties(self):
        """Load and filter heirs properties for Vance County."""
        heirs_path = self.data_dir / "raw" / "nc-hp_v2.parquet"
        print(f"Loading heirs properties from {heirs_path}")
        
        try:
            # Load the parquet file
            heirs_df = gpd.read_parquet(heirs_path)
            print(f"Initial CRS: {heirs_df.crs}")
            print(f"Initial count: {len(heirs_df)}")
            
            # Create Vance bounds in WGS84
            vance_bounds_gdf = gpd.GeoDataFrame(geometry=[self.vance_bounds], crs="EPSG:4326")
            
            # Filter for Vance County while still in WGS84
            vance_heirs = heirs_df[heirs_df.geometry.intersects(self.vance_bounds)]
            print(f"After Vance filter: {len(vance_heirs)}")
            
            # Project to NC State Plane
            vance_heirs = vance_heirs.to_crs(self.target_crs)
            print(f"Projected to: {vance_heirs.crs}")
            
            print(f"Found {len(vance_heirs)} heirs properties in Vance County")
            return vance_heirs
            
        except Exception as e:
            print(f"Error loading heirs properties: {str(e)}")
            raise
    
    def load_neighbors(self):
        """Load and filter neighboring parcels."""
        neighbors_path = self.data_dir / "raw" / "parcels_within_1_mile.parquet"
        print(f"Loading neighboring parcels from {neighbors_path}")
        
        try:
            if not neighbors_path.exists():
                print(f"Neighbors file not found: {neighbors_path}")
                return gpd.GeoDataFrame(geometry=[])
            
            # Load the parquet file
            neighbors_df = gpd.read_parquet(neighbors_path)
            print(f"Initial CRS: {neighbors_df.crs}")
            print(f"Initial count: {len(neighbors_df)}")
            
            # Create Vance bounds in WGS84
            vance_bounds_gdf = gpd.GeoDataFrame(geometry=[self.vance_bounds], crs="EPSG:4326")
            
            # Filter for Vance County while still in WGS84
            vance_neighbors = neighbors_df[neighbors_df.geometry.intersects(self.vance_bounds)]
            print(f"After Vance filter: {len(vance_neighbors)}")
            
            # Project to NC State Plane
            vance_neighbors = vance_neighbors.to_crs(self.target_crs)
            print(f"Projected to: {vance_neighbors.crs}")
            
            print(f"Found {len(vance_neighbors)} neighboring parcels in Vance County")
            return vance_neighbors
            
        except Exception as e:
            print(f"Error loading neighboring parcels: {str(e)}")
            raise
    
    def analyze_spatial_relationships(self, heirs_gdf, neighbors_gdf):
        """Analyze spatial relationships between heirs properties and neighbors."""
        # Convert areas to acres (1 sq meter = 0.000247105 acres)
        SQMETERS_TO_ACRES = 0.000247105
        
        def calculate_area_stats(gdf):
            if len(gdf) == 0:
                return {
                    'mean': None,
                    'std': None,
                    'min': None,
                    'max': None,
                    'total': None
                }
            areas = gdf.geometry.area * SQMETERS_TO_ACRES
            return {
                'mean': float(areas.mean()),
                'std': float(areas.std()),
                'min': float(areas.min()),
                'max': float(areas.max()),
                'total': float(areas.sum())
            }
        
        results = {
            'heirs_count': len(heirs_gdf),
            'neighbors_count': len(neighbors_gdf),
            'heirs_area_stats_acres': calculate_area_stats(heirs_gdf),
            'neighbor_area_stats_acres': calculate_area_stats(neighbors_gdf)
        }
        
        # Add distribution information
        if len(heirs_gdf) > 0:
            area_bins = [0, 1, 5, 10, 50, 100, float('inf')]
            areas = heirs_gdf.geometry.area * SQMETERS_TO_ACRES
            hist, _ = np.histogram(areas, bins=area_bins)
            results['heirs_area_distribution_acres'] = {
                'bins': [f"{area_bins[i]}-{area_bins[i+1]} acres" for i in range(len(area_bins)-1)],
                'counts': hist.tolist()
            }
        
        return results

    def validate_geometries(self, gdf):
        """Validate geometry quality of the GeoDataFrame."""
        validation = {
            'total_count': len(gdf),
            'valid_count': sum(gdf.geometry.is_valid),
            'invalid_count': sum(~gdf.geometry.is_valid),
            'null_count': sum(gdf.geometry.isna()),
            'geometry_types': gdf.geometry.type.value_counts().to_dict(),
            'crs': str(gdf.crs)
        }
        
        # Add column information if available
        if len(gdf) > 0:
            validation['columns'] = list(gdf.columns)
        
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