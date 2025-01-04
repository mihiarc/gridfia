#!/usr/bin/env python3

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import folium
from datetime import datetime
import logging
from shapely.ops import unary_union
import geopandas.tools

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NDVIPrototypeAnalysis:
    """Analyze NDVI patterns for the 102 prototype properties."""
    
    def __init__(self, data_dir="data"):
        """Initialize with data paths."""
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.results_dir = self.processed_dir / "prototype_analysis"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load property data first
        logger.info("Loading property data...")
        self.heirs_df = gpd.read_parquet(self.data_dir / "raw/nc-hp_v2.parquet")
        # Project to NC State Plane (meters)
        self.heirs_df = self.heirs_df.to_crs(epsg=32119)  # NC State Plane
        self.heirs_df = self.heirs_df[self.heirs_df['county_nam'].str.contains('Vance', case=False, na=False)]
        logger.info(f"Found {len(self.heirs_df)} properties in Vance County")
        
        # Load NDVI analysis results
        logger.info("Loading NDVI analysis results...")
        with open(self.processed_dir / "property_ndvi_analysis.json") as f:
            self.ndvi_data = json.load(f)
        
        # Extract properties with NDVI data
        self.ndvi_properties = []
        for prop_id, data in self.ndvi_data['heirs'].items():
            if all(year in data for year in ['2018', '2020', '2022']):
                self.ndvi_properties.append(int(prop_id))
        
        # Filter analysis dataframe
        self.analysis_df = self.heirs_df[self.heirs_df.index.isin(self.ndvi_properties)].copy()
        logger.info(f"Found {len(self.analysis_df)} properties with complete NDVI data")
        
        # Load neighbor data
        logger.info("Loading neighbor data...")
        neighbor_file = self.processed_dir / "neighbors_5.0km.parquet"
        if not neighbor_file.exists():
            logger.error(f"Neighbor file not found: {neighbor_file}")
            raise FileNotFoundError(f"Required neighbor file not found: {neighbor_file}")
        
        # Load neighbor relationships (regular parquet)
        neighbor_relationships = pd.read_parquet(neighbor_file)
        logger.info(f"Loaded {len(neighbor_relationships)} neighbor relationships")
        
        # Get unique neighbor IDs
        neighbor_ids = neighbor_relationships['neighbor_parcel_id'].unique()
        logger.info(f"Found {len(neighbor_ids)} unique neighbor parcels")
        
        # Load neighbor parcels
        logger.info("Loading neighbor parcels...")
        parcels_df = gpd.read_parquet(
            self.data_dir / "raw/nc-parcels.parquet",
            columns=['geometry', 'PARCELAPN']  # Load only necessary columns
        )
        parcels_df = parcels_df.loc[parcels_df.index.isin(neighbor_ids)]
        parcels_df = parcels_df.to_crs(epsg=32119)  # Project to NC State Plane
        logger.info(f"Loaded {len(parcels_df)} neighbor parcels")
        
        # Store the parcels dataframe
        self.neighbors_df = parcels_df
        
        # Load NDVI data for neighbors
        if 'neighbors' in self.ndvi_data:
            logger.info("Loading neighbor NDVI data...")
            self.neighbor_ndvi = []
            for idx, row in self.neighbors_df.iterrows():
                if str(idx) in self.ndvi_data['neighbors']:
                    data = self.ndvi_data['neighbors'][str(idx)]
                    if all(year in data for year in ['2018', '2020', '2022']):
                        self.neighbor_ndvi.append(idx)
            
            self.neighbors_df = self.neighbors_df[self.neighbors_df.index.isin(self.neighbor_ndvi)]
            logger.info(f"Found {len(self.neighbors_df)} neighbors with complete NDVI data")
            
            # Add relationship information
            self.neighbors_df = self.neighbors_df.join(
                neighbor_relationships.set_index('neighbor_parcel_id')[['distance_meters', 'direction']],
                how='left'
            )
            logger.info("Added neighbor relationship information")
            
            # Calculate area in acres
            self.neighbors_df['land_acre'] = self.neighbors_df.geometry.area * 0.000247105  # Convert sq meters to acres
            logger.info("Calculated land area in acres")
        
        # Create buffer for finding neighbors
        buffer_distance = 5000  # 5 km buffer
        heirs_buffer = unary_union(self.analysis_df.geometry.buffer(buffer_distance))
        
        # Find neighbors within buffer
        logger.info("Finding neighbors within buffer...")
        self.neighbors_df = self.neighbors_df[self.neighbors_df.geometry.intersects(heirs_buffer)]
        # Remove heirs properties from neighbors
        self.neighbors_df = self.neighbors_df[~self.neighbors_df.index.isin(self.heirs_df.index)]
        logger.info(f"Found {len(self.neighbors_df)} neighboring properties within {buffer_distance/1000}km")
    
    def calculate_slope(self, x, y):
        """Calculate slope using simple linear regression."""
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        denominator = sum((xi - x_mean) ** 2 for xi in x)
        return numerator / denominator if denominator != 0 else 0
    
    def calculate_temporal_stats(self):
        """Calculate temporal NDVI statistics for each property."""
        stats = []
        logger.info("Calculating temporal statistics...")
        
        # Process heirs properties
        for idx, row in self.analysis_df.iterrows():
            prop_id = str(idx)  # Convert to string for JSON lookup
            data = self.ndvi_data['heirs'][prop_id]
            
            try:
                # Get basic property info
                prop_info = {
                    'property_id': idx,  # Use integer index
                    'land_acres': float(row['land_acre']),
                    'type': 'heir'
                }
                
                # Add NDVI means for each year
                ndvi_values = []
                for year in ['2018', '2020', '2022']:
                    if year in data:
                        value = data[year]['ndvi_stats']['mean']
                        prop_info[f'ndvi_{year}'] = value
                        ndvi_values.append(value)
                
                # Calculate temporal trends
                years = [2018, 2020, 2022]
                slope = self.calculate_slope(years, ndvi_values)
                
                # Calculate changes between consecutive years
                prop_info.update({
                    'slope': slope,
                    'change_2018_2020': ndvi_values[1] - ndvi_values[0],
                    'change_2020_2022': ndvi_values[2] - ndvi_values[1]
                })
                
                stats.append(prop_info)
            except Exception as e:
                logger.warning(f"Error processing heir property {prop_id}: {str(e)}")
                continue
        
        # Process neighbor properties
        for idx, row in self.neighbors_df.iterrows():
            prop_id = str(idx)
            data = self.ndvi_data['neighbors'][prop_id]
            
            try:
                # Get basic property info
                prop_info = {
                    'property_id': idx,
                    'land_acres': float(row['land_acre']),
                    'type': 'neighbor'
                }
                
                # Add NDVI means for each year
                ndvi_values = []
                for year in ['2018', '2020', '2022']:
                    if year in data:
                        value = data[year]['ndvi_stats']['mean']
                        prop_info[f'ndvi_{year}'] = value
                        ndvi_values.append(value)
                
                # Calculate temporal trends
                years = [2018, 2020, 2022]
                slope = self.calculate_slope(years, ndvi_values)
                
                # Calculate changes between consecutive years
                prop_info.update({
                    'slope': slope,
                    'change_2018_2020': ndvi_values[1] - ndvi_values[0],
                    'change_2020_2022': ndvi_values[2] - ndvi_values[1]
                })
                
                stats.append(prop_info)
            except Exception as e:
                logger.warning(f"Error processing neighbor property {prop_id}: {str(e)}")
                continue
        
        # Convert to DataFrame
        stats_df = pd.DataFrame(stats)
        if not stats_df.empty:
            stats_df.set_index('property_id', inplace=True)
            
            # Save results
            stats_df.to_csv(self.results_dir / "temporal_stats.csv")
            
            # Generate summary statistics by property type
            summary = {
                'heirs': {
                    'mean_ndvi_by_year': {
                        year: stats_df[stats_df['type'] == 'heir'][f'ndvi_{year}'].mean()
                        for year in ['2018', '2020', '2022']
                    },
                    'std_ndvi_by_year': {
                        year: stats_df[stats_df['type'] == 'heir'][f'ndvi_{year}'].std()
                        for year in ['2018', '2020', '2022']
                    },
                    'mean_changes': {
                        '2018_2020': stats_df[stats_df['type'] == 'heir']['change_2018_2020'].mean(),
                        '2020_2022': stats_df[stats_df['type'] == 'heir']['change_2020_2022'].mean()
                    },
                    'trend_stats': {
                        'mean_slope': stats_df[stats_df['type'] == 'heir']['slope'].mean(),
                        'std_slope': stats_df[stats_df['type'] == 'heir']['slope'].std()
                    }
                },
                'neighbors': {
                    'mean_ndvi_by_year': {
                        year: stats_df[stats_df['type'] == 'neighbor'][f'ndvi_{year}'].mean()
                        for year in ['2018', '2020', '2022']
                    },
                    'std_ndvi_by_year': {
                        year: stats_df[stats_df['type'] == 'neighbor'][f'ndvi_{year}'].std()
                        for year in ['2018', '2020', '2022']
                    },
                    'mean_changes': {
                        '2018_2020': stats_df[stats_df['type'] == 'neighbor']['change_2018_2020'].mean(),
                        '2020_2022': stats_df[stats_df['type'] == 'neighbor']['change_2020_2022'].mean()
                    },
                    'trend_stats': {
                        'mean_slope': stats_df[stats_df['type'] == 'neighbor']['slope'].mean(),
                        'std_slope': stats_df[stats_df['type'] == 'neighbor']['slope'].std()
                    }
                }
            }
            
            with open(self.results_dir / "temporal_summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
            
            return stats_df, summary
        else:
            logger.error("No valid statistics could be calculated")
            return None, None
    
    def create_visualizations(self, stats_df):
        """Create visualizations of the analysis results."""
        if stats_df is None or stats_df.empty:
            logger.error("No data available for visualizations")
            return
        
        logger.info("Creating visualizations...")
        # Set up the plotting style
        plt.style.use('default')  # Use default style instead of seaborn
        
        # 1. NDVI Distribution by Year and Type
        plt.figure(figsize=(15, 6))
        years = ['2018', '2020', '2022']
        
        # Create side-by-side boxplots for heirs and neighbors
        positions = np.arange(len(years)) * 3
        width = 0.8
        
        heir_data = [stats_df[stats_df['type'] == 'heir'][f'ndvi_{year}'].dropna() for year in years]
        neighbor_data = [stats_df[stats_df['type'] == 'neighbor'][f'ndvi_{year}'].dropna() for year in years]
        
        if any(len(d) > 0 for d in heir_data + neighbor_data):
            bp1 = plt.boxplot(heir_data, positions=positions - width, tick_labels=[''] * len(years), patch_artist=True)
            bp2 = plt.boxplot(neighbor_data, positions=positions + width, tick_labels=[''] * len(years), patch_artist=True)
            
            # Set colors
            for box in bp1['boxes']:
                box.set(facecolor='lightblue')
            for box in bp2['boxes']:
                box.set(facecolor='lightgreen')
            
            plt.xticks(positions, years)
            plt.title('NDVI Distribution by Year and Property Type')
            plt.ylabel('NDVI Value')
            plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Heirs', 'Neighbors'], loc='upper right')
            plt.grid(True, alpha=0.3)
            plt.savefig(self.results_dir / "ndvi_distribution.png", dpi=300, bbox_inches='tight')
        else:
            logger.error("No valid NDVI data for distribution plot")
        plt.close()
        
        # 2. NDVI Change Over Time by Type
        plt.figure(figsize=(12, 6))
        
        # Plot heirs properties
        heir_means = [stats_df[stats_df['type'] == 'heir'][f'ndvi_{year}'].mean() for year in years]
        heir_stds = [stats_df[stats_df['type'] == 'heir'][f'ndvi_{year}'].std() for year in years]
        plt.errorbar(years, heir_means, yerr=heir_stds, marker='o', capsize=5, label='Heirs', color='blue')
        
        # Plot neighbor properties
        neighbor_means = [stats_df[stats_df['type'] == 'neighbor'][f'ndvi_{year}'].mean() for year in years]
        neighbor_stds = [stats_df[stats_df['type'] == 'neighbor'][f'ndvi_{year}'].std() for year in years]
        plt.errorbar(years, neighbor_means, yerr=neighbor_stds, marker='s', capsize=5, label='Neighbors', color='green')
        
        plt.title('Mean NDVI Change Over Time by Property Type')
        plt.ylabel('NDVI Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.results_dir / "ndvi_trend.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Property Size vs NDVI by Type
        plt.figure(figsize=(10, 6))
        heir_mask = stats_df['type'] == 'heir'
        neighbor_mask = stats_df['type'] == 'neighbor'
        
        plt.scatter(stats_df[heir_mask]['land_acres'], 
                   stats_df[heir_mask]['ndvi_2022'], 
                   alpha=0.6, label='Heirs', color='blue')
        plt.scatter(stats_df[neighbor_mask]['land_acres'], 
                   stats_df[neighbor_mask]['ndvi_2022'], 
                   alpha=0.6, label='Neighbors', color='green')
        
        plt.xlabel('Property Size (acres)')
        plt.ylabel('NDVI 2022')
        plt.title('Property Size vs NDVI Value by Type')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.results_dir / "size_vs_ndvi.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Create map of properties
        try:
            # Convert back to WGS84 for Folium
            heirs_map = self.analysis_df.copy()
            neighbors_map = self.neighbors_df.copy()
            
            # Calculate center point in projected CRS first
            center_point = heirs_map.geometry.centroid.unary_union.centroid
            center_lat = center_point.y
            center_lon = center_point.x
            
            # Now convert to WGS84
            heirs_map = heirs_map.to_crs(epsg=4326)
            neighbors_map = neighbors_map.to_crs(epsg=4326)
            
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=11,
                tiles='CartoDB positron'
            )
            
            # Create feature groups for heirs and neighbors
            heirs_group = folium.FeatureGroup(name='Heirs Properties')
            neighbors_group = folium.FeatureGroup(name='Neighboring Properties')
            
            # Add heirs properties
            for idx, row in heirs_map.iterrows():
                try:
                    # Get NDVI values and ensure they are scalar values
                    ndvi_2022 = float(stats_df.loc[idx, 'ndvi_2022']) if pd.notnull(stats_df.loc[idx, 'ndvi_2022']) else None
                    ndvi_2020 = float(stats_df.loc[idx, 'ndvi_2020']) if pd.notnull(stats_df.loc[idx, 'ndvi_2020']) else None
                    ndvi_2018 = float(stats_df.loc[idx, 'ndvi_2018']) if pd.notnull(stats_df.loc[idx, 'ndvi_2018']) else None
                    slope = float(stats_df.loc[idx, 'slope']) if pd.notnull(stats_df.loc[idx, 'slope']) else None
                    
                    # Pre-format values
                    ndvi_2022_str = f"{ndvi_2022:.3f}" if ndvi_2022 is not None else "N/A"
                    ndvi_2020_str = f"{ndvi_2020:.3f}" if ndvi_2020 is not None else "N/A"
                    ndvi_2018_str = f"{ndvi_2018:.3f}" if ndvi_2018 is not None else "N/A"
                    slope_str = f"{slope:.4f}" if slope is not None else "N/A"
                    
                    tooltip = f"""
                    <b>Heirs Property {idx}</b><br>
                    Size: {row['land_acre']:.1f} acres<br>
                    NDVI Values:<br>
                    - 2022: {ndvi_2022_str}<br>
                    - 2020: {ndvi_2020_str}<br>
                    - 2018: {ndvi_2018_str}<br>
                    Trend: {slope_str}/year
                    """
                    
                    folium.GeoJson(
                        row.geometry.__geo_interface__,
                        style_function=lambda x: {
                            'fillColor': 'blue',
                            'color': 'black',
                            'weight': 2,
                            'fillOpacity': 0.5
                        },
                        tooltip=tooltip
                    ).add_to(heirs_group)
                except Exception as e:
                    logger.warning(f"Error adding heir property {idx} to map: {str(e)}")
                    continue
            
            # Add neighbor properties
            logger.info(f"Stats DataFrame index type: {stats_df.index.dtype}")
            logger.info(f"Neighbors map index type: {neighbors_map.index.dtype}")
            logger.info(f"Number of properties in stats_df: {len(stats_df)}")
            logger.info(f"Number of properties in neighbors_map: {len(neighbors_map)}")
            logger.info(f"Sample of stats_df index: {stats_df.index[:5].tolist()}")
            logger.info(f"Sample of neighbors_map index: {neighbors_map.index[:5].tolist()}")
            
            for idx, row in neighbors_map.iterrows():
                try:
                    # Add detailed logging for debugging
                    logger.info(f"Processing neighbor property {idx}")
                    logger.info(f"Index type: {type(idx)}")
                    
                    # Test different index membership methods
                    method1 = idx in stats_df.index
                    method2 = stats_df.index.isin([idx])
                    method3 = stats_df.index.get_indexer([idx]) >= 0
                    
                    logger.info(f"Method1 (in): {method1}")
                    logger.info(f"Method2 (isin): {method2.tolist()}")
                    logger.info(f"Method3 (get_indexer): {method3.tolist()}")
                    
                    # Check if property exists in stats_df using pandas index membership
                    if not stats_df.index.isin([idx]).any():
                        logger.info(f"Skipping property {idx} - not found in stats_df")
                        continue
                        
                    # Get NDVI values and ensure they are scalar values
                    ndvi_2022 = float(stats_df.loc[idx, 'ndvi_2022']) if pd.notnull(stats_df.loc[idx, 'ndvi_2022']) else None
                    ndvi_2020 = float(stats_df.loc[idx, 'ndvi_2020']) if pd.notnull(stats_df.loc[idx, 'ndvi_2020']) else None
                    ndvi_2018 = float(stats_df.loc[idx, 'ndvi_2018']) if pd.notnull(stats_df.loc[idx, 'ndvi_2018']) else None
                    slope = float(stats_df.loc[idx, 'slope']) if pd.notnull(stats_df.loc[idx, 'slope']) else None
                    
                    # Pre-format values
                    ndvi_2022_str = f"{ndvi_2022:.3f}" if ndvi_2022 is not None else "N/A"
                    ndvi_2020_str = f"{ndvi_2020:.3f}" if ndvi_2020 is not None else "N/A"
                    ndvi_2018_str = f"{ndvi_2018:.3f}" if ndvi_2018 is not None else "N/A"
                    slope_str = f"{slope:.4f}" if slope is not None else "N/A"
                    
                    tooltip = f"""
                    <b>Neighbor Property {idx}</b><br>
                    Size: {row['land_acre']:.1f} acres<br>
                    NDVI Values:<br>
                    - 2022: {ndvi_2022_str}<br>
                    - 2020: {ndvi_2020_str}<br>
                    - 2018: {ndvi_2018_str}<br>
                    Trend: {slope_str}/year
                    """
                    
                    folium.GeoJson(
                        row.geometry.__geo_interface__,
                        style_function=lambda x: {
                            'fillColor': 'green',
                            'color': 'black',
                            'weight': 1,
                            'fillOpacity': 0.3
                        },
                        tooltip=tooltip
                    ).add_to(neighbors_group)
                except Exception as e:
                    logger.warning(f"Error adding neighbor property {idx} to map: {str(e)}")
                    continue
            
            # Add feature groups to map
            heirs_group.add_to(m)
            neighbors_group.add_to(m)
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            m.save(self.results_dir / "property_map.html")
        except Exception as e:
            logger.error(f"Error creating map: {str(e)}")
    
    def analyze_spatial_patterns(self):
        """Analyze spatial patterns in the NDVI data."""
        if self.analysis_df.empty:
            logger.error("No data available for spatial analysis")
            return None
        
        logger.info("Analyzing spatial patterns...")
        try:
            # Calculate spatial statistics using projected coordinates
            self.analysis_df['centroid'] = self.analysis_df.geometry.centroid
            self.analysis_df['x'] = self.analysis_df.centroid.x
            self.analysis_df['y'] = self.analysis_df.centroid.y
            
            # Calculate nearest neighbor distances
            from scipy.spatial import cKDTree
            coords = np.column_stack((self.analysis_df.x, self.analysis_df.y))
            tree = cKDTree(coords)
            distances, _ = tree.query(coords, k=2)  # k=2 to get nearest neighbor (first point is self)
            
            # Convert distances from meters to kilometers for readability
            distances = distances / 1000.0
            
            spatial_stats = {
                'mean_nn_distance_km': float(distances[:, 1].mean()),
                'std_nn_distance_km': float(distances[:, 1].std()),
                'min_nn_distance_km': float(distances[:, 1].min()),
                'max_nn_distance_km': float(distances[:, 1].max())
            }
            
            # Save spatial statistics
            with open(self.results_dir / "spatial_stats.json", 'w') as f:
                json.dump(spatial_stats, f, indent=2)
            
            return spatial_stats
        except Exception as e:
            logger.error(f"Error in spatial analysis: {str(e)}")
            return None
    
    def run_analysis(self):
        """Run the complete analysis pipeline."""
        logger.info("Starting analysis pipeline...")
        
        stats_df, temporal_summary = self.calculate_temporal_stats()
        if stats_df is not None:
            self.create_visualizations(stats_df)
            spatial_stats = self.analyze_spatial_patterns()
            
            # Compile complete analysis results
            results = {
                'analysis_date': datetime.now().isoformat(),
                'property_count': {
                    'heirs': len(self.analysis_df),
                    'neighbors': len(self.neighbors_df)
                },
                'temporal_summary': temporal_summary,
                'spatial_stats': spatial_stats
            }
            
            # Save complete results
            with open(self.results_dir / "analysis_results.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Analysis complete. Results saved to: {self.results_dir}")
            return results
        else:
            logger.error("Analysis failed due to data processing errors")
            return None

def main():
    """Run the prototype analysis."""
    analyzer = NDVIPrototypeAnalysis()
    results = analyzer.run_analysis()
    
    if results:
        # Print key findings
        print("\nKey Findings:")
        print(f"Properties analyzed:")
        print(f"  - Heirs: {results['property_count']['heirs']}")
        print(f"  - Neighbors: {results['property_count']['neighbors']}")
        
        print("\nTemporal Trends:")
        print("Heirs Properties:")
        for year, mean_ndvi in results['temporal_summary']['heirs']['mean_ndvi_by_year'].items():
            print(f"  {year} mean NDVI: {mean_ndvi:.4f}")
        print(f"  Mean slope: {results['temporal_summary']['heirs']['trend_stats']['mean_slope']:.4f}")
        
        print("\nNeighboring Properties:")
        for year, mean_ndvi in results['temporal_summary']['neighbors']['mean_ndvi_by_year'].items():
            print(f"  {year} mean NDVI: {mean_ndvi:.4f}")
        print(f"  Mean slope: {results['temporal_summary']['neighbors']['trend_stats']['mean_slope']:.4f}")
        
        if results['spatial_stats']:
            print(f"\nMean nearest neighbor distance: {results['spatial_stats']['mean_nn_distance_km']:.4f} km")
    else:
        print("\nAnalysis failed. Check the logs for details.")

if __name__ == '__main__':
    main() 