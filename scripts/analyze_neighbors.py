#!/usr/bin/env python3

import pandas as pd
import geopandas as gpd
from pathlib import Path
import json
from shapely.geometry import box
import numpy as np
from rtree import index
from tqdm import tqdm

class NeighborAnalyzer:
    """Analyzes spatial relationships between heirs properties and potential neighbors."""
    
    def __init__(self, data_dir="../data", buffer_distances=[1000, 2500, 5000, 7500, 10000]):
        """
        Initialize the analyzer with configurable buffer distances (in meters).
        Default distances: 1km, 2.5km, 5km, 7.5km, 10km (approximately 0.6-6.2 miles)
        """
        self.data_dir = Path(data_dir)
        self.buffer_distances = buffer_distances
        # NC State Plane (meters)
        self.target_crs = "EPSG:2264"
        # Vance County bounds from NDVI analysis (in WGS84)
        self.vance_bounds = box(
            -78.51124666236885,  # west
            36.16363110639788,   # south
            -78.27717264878582,  # east
            36.24948309810118    # north
        )
    
    def load_and_prepare_data(self):
        """Load and prepare heirs properties and NC parcels data."""
        print("\nLoading datasets...")
        
        # Load heirs properties
        heirs_path = self.data_dir / "raw" / "nc-hp_v2.parquet"
        print(f"Loading heirs properties from {heirs_path}")
        heirs_df = gpd.read_parquet(heirs_path)
        
        # Load NC parcels
        parcels_path = self.data_dir / "raw" / "nc-parcels.parquet"
        print(f"Loading NC parcels from {parcels_path}")
        parcels_df = gpd.read_parquet(parcels_path)
        
        # Create Vance bounds in target CRS
        vance_bounds_gdf = gpd.GeoDataFrame(
            geometry=[self.vance_bounds], 
            crs="EPSG:4326"
        ).to_crs(self.target_crs)
        vance_bounds_geom = vance_bounds_gdf.geometry.iloc[0]
        
        # Filter and project heirs properties
        print("\nProcessing heirs properties...")
        vance_heirs = heirs_df[heirs_df.geometry.intersects(self.vance_bounds)]
        vance_heirs = vance_heirs.to_crs(self.target_crs)
        print(f"Found {len(vance_heirs)} heirs properties in Vance County")
        
        # Filter and project NC parcels
        print("\nProcessing NC parcels...")
        vance_parcels = parcels_df[parcels_df.geometry.intersects(self.vance_bounds)]
        vance_parcels = vance_parcels.to_crs(self.target_crs)
        print(f"Found {len(vance_parcels)} total parcels in Vance County")
        
        return vance_heirs, vance_parcels
    
    def create_spatial_index(self, gdf):
        """Create an R-tree spatial index for the GeoDataFrame."""
        idx = index.Index()
        for pos, row in enumerate(gdf.itertuples()):
            if row.geometry.is_valid:
                idx.insert(pos, row.geometry.bounds)
        return idx
    
    def find_neighbors(self, heirs_gdf, parcels_gdf, buffer_distance):
        """
        Find neighbors for heirs properties within specified buffer distance.
        Returns a DataFrame with neighbor relationships and metrics.
        """
        print(f"\nFinding neighbors within {buffer_distance/1000:.1f}km...")
        
        # Create spatial index for parcels
        print("Creating spatial index...")
        spatial_index = self.create_spatial_index(parcels_gdf)
        
        neighbor_records = []
        
        # Process each heirs property
        for heir_idx, heir_row in tqdm(heirs_gdf.iterrows(), 
                                     total=len(heirs_gdf),
                                     desc="Processing heirs properties"):
            # Create buffer
            heir_buffer = heir_row.geometry.buffer(buffer_distance)
            
            # Find potential neighbors using spatial index
            potential_idxs = list(spatial_index.intersection(heir_buffer.bounds))
            potential_neighbors = parcels_gdf.iloc[potential_idxs]
            
            # Filter actual intersections and calculate metrics
            for neighbor_idx, neighbor_row in potential_neighbors.iterrows():
                if (neighbor_row.geometry.intersects(heir_buffer) and 
                    not neighbor_row.geometry.equals(heir_row.geometry)):
                    
                    # Calculate distance and direction
                    distance = heir_row.geometry.distance(neighbor_row.geometry)
                    centroid_vector = (
                        neighbor_row.geometry.centroid.x - heir_row.geometry.centroid.x,
                        neighbor_row.geometry.centroid.y - heir_row.geometry.centroid.y
                    )
                    direction = self.calculate_direction(centroid_vector)
                    
                    # Calculate shared boundary if any
                    try:
                        shared_boundary = heir_row.geometry.intersection(
                            neighbor_row.geometry
                        ).length
                    except:
                        shared_boundary = 0
                    
                    neighbor_records.append({
                        'heirs_parcel_id': heir_idx,
                        'neighbor_parcel_id': neighbor_idx,
                        'distance_meters': distance,
                        'shared_boundary_meters': shared_boundary,
                        'neighbor_area_acres': neighbor_row.geometry.area * 0.000247105,
                        'direction': direction,
                        'buffer_distance': buffer_distance
                    })
        
        return pd.DataFrame(neighbor_records)
    
    @staticmethod
    def calculate_direction(vector):
        """Calculate cardinal direction based on vector between centroids."""
        x, y = vector
        angle = np.degrees(np.arctan2(y, x))
        
        if -22.5 <= angle <= 22.5:
            return 'E'
        elif 22.5 < angle <= 67.5:
            return 'NE'
        elif 67.5 < angle <= 112.5:
            return 'N'
        elif 112.5 < angle <= 157.5:
            return 'NW'
        elif angle > 157.5 or angle <= -157.5:
            return 'W'
        elif -157.5 < angle <= -112.5:
            return 'SW'
        elif -112.5 < angle <= -67.5:
            return 'S'
        else:
            return 'SE'
    
    def analyze_neighbor_distribution(self, neighbor_df):
        """Analyze the distribution of neighbors."""
        stats = {
            'total_relationships': len(neighbor_df),
            'unique_heirs_parcels': neighbor_df['heirs_parcel_id'].nunique(),
            'unique_neighbor_parcels': neighbor_df['neighbor_parcel_id'].nunique(),
            'distance_stats': {
                'min': float(neighbor_df['distance_meters'].min()),
                'max': float(neighbor_df['distance_meters'].max()),
                'mean': float(neighbor_df['distance_meters'].mean()),
                'median': float(neighbor_df['distance_meters'].median()),
                'std': float(neighbor_df['distance_meters'].std())
            },
            'direction_distribution': neighbor_df['direction'].value_counts().to_dict(),
            'neighbors_per_heir': {
                'min': int(neighbor_df.groupby('heirs_parcel_id').size().min()),
                'max': int(neighbor_df.groupby('heirs_parcel_id').size().max()),
                'mean': float(neighbor_df.groupby('heirs_parcel_id').size().mean()),
                'median': float(neighbor_df.groupby('heirs_parcel_id').size().median())
            }
        }
        return stats

def main():
    """Main analysis function."""
    analyzer = NeighborAnalyzer()
    
    # Load and prepare data
    heirs_gdf, parcels_gdf = analyzer.load_and_prepare_data()
    
    # Process each buffer distance
    results = {}
    for distance in analyzer.buffer_distances:
        # Find neighbors
        neighbors_df = analyzer.find_neighbors(heirs_gdf, parcels_gdf, distance)
        
        # Analyze distribution
        stats = analyzer.analyze_neighbor_distribution(neighbors_df)
        
        # Store results
        results[f"{distance/1000:.1f}km"] = {
            'statistics': stats,
            'neighbors_file': f"neighbors_{distance/1000:.1f}km.parquet"
        }
        
        # Save neighbor relationships
        output_path = Path("../data/processed") / results[f"{distance/1000:.1f}km"]['neighbors_file']
        output_path.parent.mkdir(parents=True, exist_ok=True)
        neighbors_df.to_parquet(output_path)
    
    # Save analysis results
    with open("../data/processed/neighbor_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nAnalysis complete. Results saved to:")
    print("- Neighbor relationships: ../data/processed/neighbors_*.parquet")
    print("- Analysis results: ../data/processed/neighbor_analysis_results.json")

if __name__ == '__main__':
    main() 