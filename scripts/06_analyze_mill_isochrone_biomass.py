#!/usr/bin/env python
"""
Analyze forest biomass within 120-minute drive time isochrone of proposed timber mill.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import zarr
import zarr.storage
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import Point
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import pyproj

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bigmap.utils.montana_config import load_montana_config

console = Console()


def analyze_mill_isochrone_biomass():
    """
    Calculate biomass within mill isochrone area.
    """
    # Load configuration
    config = load_montana_config()
    
    console.print("[bold cyan]Analyzing Forest Biomass for Timber Mill Site[/bold cyan]")
    
    # Load mill location
    mill_csv = Path("config/montana_timber_mill_location.csv")
    mill_df = pd.read_csv(mill_csv)
    mill_site = mill_df[mill_df['type'] == 'mill'].iloc[0]
    
    console.print(f"\n[cyan]Mill Location:[/cyan]")
    console.print(f"  Name: {mill_site['name']}")
    console.print(f"  Coordinates: ({mill_site['lat']:.6f}, {mill_site['lon']:.6f})")
    console.print(f"  Notes: {mill_site['notes']}")
    
    # Load 120-minute isochrone
    isochrone_path = Path("output/mill_isochrone_120min.geojson")
    console.print(f"\n[cyan]Loading isochrone data from:[/cyan] {isochrone_path}")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Loading isochrone...", total=None)
        isochrone_gdf = gpd.read_file(isochrone_path)
        progress.update(task, completed=True)
    
    # Get the single isochrone record
    if len(isochrone_gdf) == 0:
        console.print("[red]Error: No isochrone found in file[/red]")
        return
    mill_isochrone = isochrone_gdf.iloc[0]
    isochrone_geom = mill_isochrone.geometry
    
    console.print(f"  Isochrone area: {isochrone_geom.area / 1e6:.2f} km²")
    
    # Transform isochrone to Montana State Plane
    transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:2256', always_xy=True)
    
    # Transform the geometry
    if isochrone_geom.geom_type == 'Polygon':
        coords = list(isochrone_geom.exterior.coords)
        transformed_coords = [transformer.transform(x, y) for x, y in coords]
        from shapely.geometry import Polygon
        isochrone_state_plane = Polygon(transformed_coords)
    else:
        # MultiPolygon
        from shapely.geometry import MultiPolygon, Polygon
        transformed_polygons = []
        for poly in isochrone_geom.geoms:
            coords = list(poly.exterior.coords)
            transformed_coords = [transformer.transform(x, y) for x, y in coords]
            transformed_polygons.append(Polygon(transformed_coords))
        isochrone_state_plane = MultiPolygon(transformed_polygons)
    
    # Load zarr store
    zarr_path = Path(str(config.zarr_output_path).replace('.zarr', '_clipped.zarr'))
    console.print(f"\n[cyan]Loading forest data from:[/cyan] {zarr_path}")
    
    store = zarr.storage.LocalStore(zarr_path)
    root = zarr.open_group(store=store, mode='r')
    
    biomass = root['biomass']
    layer_codes = root['layer_codes']
    layer_names = root['layer_names']
    bounds = root.attrs['bounds']
    
    # Create affine transform for the zarr data
    height, width = biomass.shape[1], biomass.shape[2]
    pixel_width = (bounds[2] - bounds[0]) / width
    pixel_height = (bounds[3] - bounds[1]) / height
    
    from rasterio.transform import from_bounds
    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)
    
    # Create mask for isochrone area
    console.print("\n[cyan]Creating spatial mask for isochrone area...[/cyan]")
    
    # Geometry mask returns True for pixels outside the geometry, so we invert it
    mask = ~geometry_mask([isochrone_state_plane], transform=transform, 
                         out_shape=(height, width))
    
    # Calculate area statistics
    pixel_area_sqft = pixel_width * pixel_height
    pixel_area_ha = pixel_area_sqft * 0.0000092903  # sq ft to hectares
    
    total_pixels = np.sum(mask)
    total_area_ha = total_pixels * pixel_area_ha
    
    console.print(f"\n[cyan]Isochrone Coverage:[/cyan]")
    console.print(f"  Pixels within isochrone: {total_pixels:,}")
    console.print(f"  Area: {total_area_ha:,.0f} hectares ({total_area_ha * 2.47105:,.0f} acres)")
    
    # Analyze biomass for each layer
    console.print("\n[cyan]Calculating biomass statistics...[/cyan]")
    
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        # Analyze species layers and timber
        species_indices = list(range(config.species_start_idx, config.species_end_idx + 1))
        for layer_idx in species_indices + [config.timber_idx]:  # Individual species, then timber
            task = progress.add_task(
                f"Processing {layer_names[layer_idx]}...", 
                total=None
            )
            
            # Load layer data
            layer_data = biomass[layer_idx, :, :]
            
            # Apply mask
            masked_data = layer_data * mask
            
            # Calculate statistics
            non_zero_pixels = np.sum((masked_data > 0) & mask)
            forest_area_ha = non_zero_pixels * pixel_area_ha
            
            # Total biomass (data is in Mg/ha)
            total_biomass_mg = np.sum(masked_data) * pixel_area_ha
            
            # Mean biomass (only where present)
            if non_zero_pixels > 0:
                mean_biomass = np.sum(masked_data) / non_zero_pixels
            else:
                mean_biomass = 0
            
            results.append({
                'layer_idx': layer_idx,
                'code': str(layer_codes[layer_idx]),
                'name': str(layer_names[layer_idx]),
                'forest_pixels': non_zero_pixels,
                'forest_area_ha': forest_area_ha,
                'total_biomass_mg': total_biomass_mg,
                'mean_biomass_mg_ha': mean_biomass,
                'coverage_pct': (non_zero_pixels / total_pixels * 100) if total_pixels > 0 else 0
            })
            
            progress.update(task, completed=True)
    
    # Create results table
    table = Table(title="\nBiomass Within 120-Minute Drive Time")
    table.add_column("Species", style="green")
    table.add_column("Code", style="yellow")
    table.add_column("Forest Area (ha)", justify="right")
    table.add_column("Total Biomass (Mg)", justify="right")
    table.add_column("Mean (Mg/ha)", justify="right")
    table.add_column("Coverage %", justify="right")
    
    # Sort results to show individual species first, then timber
    sorted_results = sorted(results, key=lambda x: (0 if x['layer_idx'] in species_indices else 1, 
                                                   x['layer_idx']))
    
    for result in sorted_results:
        if result['layer_idx'] == config.timber_idx:  # Add separator before timber
            table.add_row("─" * 20, "─" * 4, "─" * 12, "─" * 15, "─" * 10, "─" * 8)
        
        table.add_row(
            result['name'],
            result['code'],
            f"{result['forest_area_ha']:,.0f}",
            f"{result['total_biomass_mg']:,.0f}",
            f"{result['mean_biomass_mg_ha']:.1f}",
            f"{result['coverage_pct']:.1f}%"
        )
    
    console.print(table)
    
    # Calculate supply metrics
    console.print("\n[cyan]Supply Accessibility Metrics:[/cyan]")
    
    timber_result = next(r for r in results if r['layer_idx'] == config.timber_idx)
    
    # Annual sustainable harvest estimate (conservative 1-2% of standing biomass)
    sustainable_harvest_low = timber_result['total_biomass_mg'] * 0.01
    sustainable_harvest_high = timber_result['total_biomass_mg'] * 0.02
    
    console.print(f"  Total timber biomass: {timber_result['total_biomass_mg']:,.0f} Mg")
    console.print(f"  Estimated sustainable annual harvest: {sustainable_harvest_low:,.0f} - {sustainable_harvest_high:,.0f} Mg/year")
    console.print(f"  Average timber density: {timber_result['mean_biomass_mg_ha']:.1f} Mg/ha")
    console.print(f"  Forested area within 2-hour drive: {timber_result['forest_area_ha']:,.0f} ha")
    
    # Save results as markdown report
    # Save to output root as requested
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame(sorted_results)
    
    # Create markdown report
    report_path = output_dir / "mill_site_analysis_report.md"
    with open(report_path, 'w') as f:
        f.write("# Montana Timber Mill Site Analysis Report\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write(f"**Analysis Date:** {pd.Timestamp.now().strftime('%B %d, %Y')}\n\n")
        f.write(f"**Mill Location:** {mill_site['name']}\n")
        f.write(f"**Coordinates:** {mill_site['lat']:.6f}°N, {mill_site['lon']:.6f}°W\n")
        f.write(f"**Notes:** {mill_site['notes']}\n\n")
        
        f.write("### Key Findings\n\n")
        f.write(f"- **Total Timber Biomass:** {timber_result['total_biomass_mg']:,.0f} Mg\n")
        f.write(f"- **Forest Coverage:** {timber_result['coverage_pct']:.1f}% of 120-minute drive area\n")
        
        f.write("## 120-Minute Drive Time Analysis\n\n")
        f.write("### Coverage Area\n\n")
        f.write(f"- **Total Area:** {total_area_ha:,.0f} hectares ({total_area_ha * 2.47105:,.0f} acres)\n")
        f.write(f"- **Forested Area:** {timber_result['forest_area_ha']:,.0f} hectares ({timber_result['forest_area_ha'] * 2.47105:,.0f} acres)\n")
        f.write(f"- **Forest Coverage:** {timber_result['coverage_pct']:.1f}%\n\n")
        
        f.write("### Biomass Summary\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Total Timber Biomass | {timber_result['total_biomass_mg']:,.0f} Mg |\n")
        f.write(f"| Average Density | {timber_result['mean_biomass_mg_ha']:.1f} Mg/ha |\n")
        
        f.write("## Species Composition\n\n")
        f.write("### Biomass by Species\n\n")
        f.write("| Species | Code | Forest Area (ha) | Total Biomass (Mg) | Mean Density (Mg/ha) | % of Total |\n")
        f.write("|---------|------|------------------|-------------------|---------------------|------------|\n")
        
        for result in sorted_results:
            if result['layer_idx'] in species_indices:
                pct_of_total = result['total_biomass_mg']/timber_result['total_biomass_mg']*100
                f.write(f"| {result['name']} | {result['code']} | ")
                f.write(f"{result['forest_area_ha']:,.0f} | ")
                f.write(f"{result['total_biomass_mg']:,.0f} | ")
                f.write(f"{result['mean_biomass_mg_ha']:.1f} | ")
                f.write(f"{pct_of_total:.1f}% |\n")
        
        f.write(f"| **Total (5 Species)** | **{timber_result['code']}** | ")
        f.write(f"**{timber_result['forest_area_ha']:,.0f}** | ")
        f.write(f"**{timber_result['total_biomass_mg']:,.0f}** | ")
        f.write(f"**{timber_result['mean_biomass_mg_ha']:.1f}** | ")
        f.write(f"**100.0%** |\n\n")
        
        f.write("### Species Distribution\n\n")
        for result in sorted_results:
            if result['layer_idx'] in species_indices:
                pct_of_total = result['total_biomass_mg']/timber_result['total_biomass_mg']*100
                f.write(f"- **{result['name']}**: {pct_of_total:.1f}% of total timber biomass\n")
        
        f.write("\n## Supply Accessibility Metrics\n\n")
        f.write(f"The analysis shows that within a 120-minute drive of the proposed mill site, ")
        f.write(f"there are approximately **{timber_result['total_biomass_mg']:,.0f} Mg** of standing timber biomass ")
        f.write(f"across **{timber_result['forest_area_ha']:,.0f} hectares** of forested land.\n\n")
        
        f.write("---\n\n")
        f.write("*Report generated by Forest Service R&D: Forest Economics & Policy Unit*\n")
    
    console.print(f"\n[green]✓ Markdown report saved to:[/green] {report_path}")
    
    # Return results for visualization
    return {
        'mill_site': mill_site,
        'isochrone_geom': isochrone_state_plane,
        'mask': mask,
        'results': results_df,
        'report_path': report_path,
        'bounds': bounds,
        'transform': transform
    }


if __name__ == "__main__":
    analyze_mill_isochrone_biomass()