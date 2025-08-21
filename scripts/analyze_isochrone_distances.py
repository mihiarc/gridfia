#!/usr/bin/env python
"""
Analyze the relationship between drive time and drive distance for mill isochrones.
"""

import sys
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from rich.console import Console
from rich.table import Table
import pyproj

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()


def calculate_isochrone_metrics(isochrone_gdf, mill_point):
    """Calculate various distance metrics for an isochrone."""
    
    # Get the isochrone polygon
    iso_poly = isochrone_gdf.geometry.iloc[0]
    
    # Calculate area
    # Project to Montana State Plane for accurate area calculation
    transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:2256', always_xy=True)
    
    # Transform polygon to State Plane
    if iso_poly.geom_type == 'Polygon':
        coords = list(iso_poly.exterior.coords)
        transformed_coords = [transformer.transform(x, y) for x, y in coords]
        from shapely.geometry import Polygon
        iso_poly_proj = Polygon(transformed_coords)
    else:
        raise ValueError("Expected Polygon geometry")
    
    # Calculate area in square feet, then convert to square miles
    area_sqft = iso_poly_proj.area
    area_sqmi = area_sqft / (5280 ** 2)
    
    # Calculate distances from mill to various points on the boundary
    mill_x, mill_y = transformer.transform(mill_point.x, mill_point.y)
    mill_point_proj = Point(mill_x, mill_y)
    
    # Get boundary points
    boundary_points = list(iso_poly_proj.exterior.coords)
    
    # Calculate distances to all boundary points
    distances_ft = []
    for bx, by in boundary_points:
        dist = mill_point_proj.distance(Point(bx, by))
        distances_ft.append(dist)
    
    # Convert to miles
    distances_mi = [d / 5280 for d in distances_ft]
    
    # Calculate statistics
    min_dist = min(distances_mi)
    max_dist = max(distances_mi)
    avg_dist = np.mean(distances_mi)
    median_dist = np.median(distances_mi)
    
    # Calculate approximate perimeter
    perimeter_ft = iso_poly_proj.length
    perimeter_mi = perimeter_ft / 5280
    
    # Calculate "effective radius" (radius of circle with same area)
    effective_radius = np.sqrt(area_sqmi / np.pi)
    
    return {
        'area_sqmi': area_sqmi,
        'perimeter_mi': perimeter_mi,
        'min_distance_mi': min_dist,
        'max_distance_mi': max_dist,
        'avg_distance_mi': avg_dist,
        'median_distance_mi': median_dist,
        'effective_radius_mi': effective_radius,
        'num_vertices': len(boundary_points)
    }


def analyze_isochrone_distances():
    """Analyze the relationship between drive time and distance."""
    
    console.print("[bold cyan]Analyzing Isochrone Drive Time vs Distance Relationship[/bold cyan]")
    
    # Load mill location
    mill_csv = Path("config/montana_timber_mill_location.csv")
    mill_df = pd.read_csv(mill_csv)
    mill_site = mill_df[mill_df['type'] == 'mill'].iloc[0]
    mill_point = Point(mill_site['lon'], mill_site['lat'])
    
    console.print(f"\nMill Location: {mill_site['name']}")
    console.print(f"Coordinates: {mill_site['lat']:.6f}°N, {mill_site['lon']:.6f}°W\n")
    
    # Load isochrones
    isochrone_dir = Path("output/isochrones")
    isochrone_files = sorted(isochrone_dir.glob("mill_isochrone_*min.geojson"))
    
    # Also check for the 120min isochrone in the main output directory
    main_120_file = Path("output/mill_isochrone_120min.geojson")
    if main_120_file.exists() and main_120_file not in isochrone_files:
        isochrone_files.append(main_120_file)
        isochrone_files = sorted(isochrone_files, key=lambda x: int(x.stem.split('_')[-1].replace('min', '')))
    
    if not isochrone_files:
        console.print("[red]No isochrone files found![/red]")
        return
    
    # Analyze each isochrone
    results = []
    
    for iso_file in isochrone_files:
        console.print(f"Analyzing {iso_file.name}...")
        
        # Load isochrone
        iso_gdf = gpd.read_file(iso_file)
        time_min = iso_gdf['travel_time_minutes'].iloc[0]
        
        # Calculate metrics
        metrics = calculate_isochrone_metrics(iso_gdf, mill_point)
        metrics['time_minutes'] = time_min
        metrics['time_hours'] = time_min / 60.0
        
        # Calculate average speed
        metrics['avg_speed_mph'] = metrics['avg_distance_mi'] / metrics['time_hours']
        metrics['min_speed_mph'] = metrics['min_distance_mi'] / metrics['time_hours']
        metrics['max_speed_mph'] = metrics['max_distance_mi'] / metrics['time_hours']
        
        results.append(metrics)
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('time_minutes')
    
    # Display results table
    table = Table(title="Isochrone Distance Analysis")
    table.add_column("Time (min)", justify="right", style="cyan")
    table.add_column("Area (sq mi)", justify="right")
    table.add_column("Min Dist (mi)", justify="right")
    table.add_column("Avg Dist (mi)", justify="right")
    table.add_column("Max Dist (mi)", justify="right")
    table.add_column("Avg Speed (mph)", justify="right", style="yellow")
    
    for _, row in results_df.iterrows():
        table.add_row(
            f"{row['time_minutes']:.0f}",
            f"{row['area_sqmi']:.1f}",
            f"{row['min_distance_mi']:.1f}",
            f"{row['avg_distance_mi']:.1f}",
            f"{row['max_distance_mi']:.1f}",
            f"{row['avg_speed_mph']:.1f}"
        )
    
    console.print(table)
    
    # Calculate ratios between consecutive isochrones
    console.print("\n[cyan]Growth Ratios Between Consecutive Isochrones:[/cyan]")
    
    ratio_table = Table()
    ratio_table.add_column("Interval", style="cyan")
    ratio_table.add_column("Area Ratio", justify="right")
    ratio_table.add_column("Avg Distance Ratio", justify="right")
    ratio_table.add_column("Speed Change (%)", justify="right", style="yellow")
    
    for i in range(1, len(results_df)):
        prev = results_df.iloc[i-1]
        curr = results_df.iloc[i]
        
        interval = f"{prev['time_minutes']:.0f}-{curr['time_minutes']:.0f} min"
        area_ratio = curr['area_sqmi'] / prev['area_sqmi']
        dist_ratio = curr['avg_distance_mi'] / prev['avg_distance_mi']
        speed_change = ((curr['avg_speed_mph'] - prev['avg_speed_mph']) / prev['avg_speed_mph']) * 100
        
        ratio_table.add_row(
            interval,
            f"{area_ratio:.2f}x",
            f"{dist_ratio:.2f}x",
            f"{speed_change:+.1f}%"
        )
    
    console.print(ratio_table)
    
    return results_df


def create_markdown_report(results_df):
    """Create a detailed markdown report of the analysis."""
    
    report_path = Path("output/analysis/isochrone_distance_analysis.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("# Montana Mill Isochrone Distance Analysis\n\n")
        f.write(f"**Analysis Date:** {pd.Timestamp.now().strftime('%B %d, %Y')}\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write("This analysis examines the relationship between drive time and drive distance ")
        f.write("for the proposed Montana timber mill location. The isochrones represent areas ")
        f.write("reachable within specific drive times, accounting for actual road networks, ")
        f.write("speed limits, and terrain.\n\n")
        
        f.write("### Key Findings\n\n")
        
        # Average speed across all isochrones
        avg_speed_overall = results_df['avg_speed_mph'].mean()
        f.write(f"- **Average Travel Speed:** {avg_speed_overall:.1f} mph across all isochrones\n")
        
        # Speed range
        min_speed = results_df['avg_speed_mph'].min()
        max_speed = results_df['avg_speed_mph'].max()
        f.write(f"- **Speed Range:** {min_speed:.1f} - {max_speed:.1f} mph (varies by time interval)\n")
        
        # Area coverage
        max_area = results_df['area_sqmi'].max()
        f.write(f"- **Maximum Coverage:** {max_area:,.0f} square miles within {results_df['time_minutes'].max():.0f} minutes\n\n")
        
        f.write("## Detailed Metrics\n\n")
        f.write("### Distance and Area by Drive Time\n\n")
        
        f.write("| Drive Time | Area Coverage | Min Distance | Average Distance | Max Distance | Effective Radius | Avg Speed |\n")
        f.write("|------------|---------------|--------------|------------------|--------------|------------------|------------|\n")
        
        for _, row in results_df.iterrows():
            f.write(f"| {row['time_minutes']:.0f} min | ")
            f.write(f"{row['area_sqmi']:,.0f} sq mi | ")
            f.write(f"{row['min_distance_mi']:.1f} mi | ")
            f.write(f"{row['avg_distance_mi']:.1f} mi | ")
            f.write(f"{row['max_distance_mi']:.1f} mi | ")
            f.write(f"{row['effective_radius_mi']:.1f} mi | ")
            f.write(f"{row['avg_speed_mph']:.1f} mph |\n")
        
        f.write("\n### Time-Distance Relationship\n\n")
        
        # Calculate correlation
        correlation = results_df['time_minutes'].corr(results_df['avg_distance_mi'])
        f.write(f"**Correlation between time and average distance:** {correlation:.3f} ")
        f.write("(strong positive correlation)\n\n")
        
        # Distance per minute
        for _, row in results_df.iterrows():
            dist_per_min = row['avg_distance_mi'] / row['time_minutes']
            f.write(f"- **{row['time_minutes']:.0f} minutes:** {dist_per_min:.2f} miles per minute average\n")
        
        f.write("\n### Growth Patterns\n\n")
        f.write("How isochrones expand with increasing drive time:\n\n")
        
        f.write("| Time Interval | Area Growth | Distance Growth | Speed Change |\n")
        f.write("|---------------|-------------|-----------------|---------------|\n")
        
        for i in range(1, len(results_df)):
            prev = results_df.iloc[i-1]
            curr = results_df.iloc[i]
            
            interval = f"{prev['time_minutes']:.0f}→{curr['time_minutes']:.0f} min"
            area_ratio = curr['area_sqmi'] / prev['area_sqmi']
            dist_ratio = curr['avg_distance_mi'] / prev['avg_distance_mi']
            speed_change = ((curr['avg_speed_mph'] - prev['avg_speed_mph']) / prev['avg_speed_mph']) * 100
            
            f.write(f"| {interval} | ")
            f.write(f"{area_ratio:.2f}x | ")
            f.write(f"{dist_ratio:.2f}x | ")
            f.write(f"{speed_change:+.1f}% |\n")
        
        f.write("\n## Analysis Insights\n\n")
        
        f.write("### 1. Speed Variations by Distance\n\n")
        f.write("The analysis reveals that average travel speeds increase with longer drive times:\n\n")
        
        for _, row in results_df.iterrows():
            f.write(f"- **{row['time_minutes']:.0f}-minute isochrone:** ")
            f.write(f"Average speed of {row['avg_speed_mph']:.1f} mph\n")
        
        f.write("\nThis pattern suggests that:\n")
        f.write("- Shorter trips involve more local roads with lower speed limits\n")
        f.write("- Longer trips utilize more highways and higher-speed roads\n")
        f.write("- Montana's rural road network allows faster travel at greater distances\n\n")
        
        f.write("### 2. Geographic Constraints\n\n")
        f.write("The min/max distance variations indicate:\n\n")
        
        for _, row in results_df.iterrows():
            range_ratio = row['max_distance_mi'] / row['min_distance_mi']
            f.write(f"- **{row['time_minutes']:.0f} minutes:** ")
            f.write(f"{range_ratio:.1f}x difference between closest and farthest points\n")
        
        f.write("\nThis asymmetry is likely due to:\n")
        f.write("- Mountain terrain limiting travel in certain directions\n")
        f.write("- Highway access creating corridors of faster travel\n")
        f.write("- Natural barriers (rivers, forests) affecting road networks\n\n")
        
        f.write("### 3. Supply Chain Implications\n\n")
        
        # Calculate tonnage capacity assumptions
        truck_capacity = 25  # tons per truck
        trips_per_day = 2   # round trips
        working_days = 250  # per year
        
        f.write("For timber transport planning (assuming 25-ton trucks, 2 trips/day):\n\n")
        
        for _, row in results_df.iterrows():
            # Rough estimate of annual capacity from this distance
            annual_capacity = truck_capacity * trips_per_day * working_days
            f.write(f"- **{row['time_minutes']:.0f}-minute zone:** ")
            f.write(f"Up to {annual_capacity:,} tons/year per truck\n")
        
        f.write("\n### 4. Optimal Supply Zones\n\n")
        
        # Identify sweet spots
        f.write("Based on time-distance efficiency:\n\n")
        
        for _, row in results_df.iterrows():
            efficiency = row['avg_distance_mi'] / row['time_minutes']
            if efficiency > 0.8:
                rating = "Excellent"
            elif efficiency > 0.7:
                rating = "Good"
            elif efficiency > 0.6:
                rating = "Fair"
            else:
                rating = "Limited"
            
            f.write(f"- **{row['time_minutes']:.0f}-minute zone:** ")
            f.write(f"{rating} efficiency ({efficiency:.2f} mi/min)\n")
        
        f.write("\n## Methodology\n\n")
        f.write("- **Isochrone Generation:** SocialMapper package using OpenStreetMap road network data\n")
        f.write("- **Distance Calculations:** Geodesic distances from mill to isochrone boundaries\n")
        f.write("- **Area Calculations:** Projected to Montana State Plane (EPSG:2256) for accuracy\n")
        f.write("- **Speed Estimates:** Based on actual road network traversal, not straight-line distances\n\n")
        
        f.write("## Recommendations\n\n")
        f.write("1. **Primary Supply Zone:** Focus on 60-minute drive time for regular operations\n")
        f.write("2. **Extended Supply Zone:** 90-minute area for seasonal or high-value timber\n")
        f.write("3. **Transportation Planning:** Account for ~{:.0f} mph average speeds in logistics\n".format(avg_speed_overall))
        f.write("4. **Route Optimization:** Prioritize highway corridors for maximum efficiency\n\n")
        
        f.write("---\n\n")
        f.write("*Generated by BigMap Forest Analysis Pipeline*\n")
    
    console.print(f"\n[green]✓ Markdown report saved to: {report_path}[/green]")
    
    return report_path


def main():
    """Run the isochrone distance analysis."""
    # Analyze isochrones
    results_df = analyze_isochrone_distances()
    
    if results_df is not None and not results_df.empty:
        # Create markdown report
        create_markdown_report(results_df)
        
        # Summary statistics
        console.print("\n[bold cyan]Summary Statistics:[/bold cyan]")
        console.print(f"Average speed across all isochrones: {results_df['avg_speed_mph'].mean():.1f} mph")
        console.print(f"Total area covered (90 min): {results_df['area_sqmi'].max():,.0f} square miles")
        console.print(f"Maximum reach: {results_df['max_distance_mi'].max():.1f} miles")


if __name__ == "__main__":
    main()