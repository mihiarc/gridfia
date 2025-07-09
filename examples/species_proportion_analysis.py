#!/usr/bin/env python3
"""
Example: Species Proportion Analysis

This script demonstrates how to calculate species proportions and percentages
using the BigMap calculation framework.
"""

from pathlib import Path
import numpy as np
import zarr
from rich.console import Console
from rich.table import Table

from bigmap.config import BigMapSettings, CalculationConfig
from bigmap.core.forest_metrics import ForestMetricsProcessor
from bigmap.core.calculations import (
    SpeciesProportion, 
    SpeciesGroupProportion, 
    SpeciesPercentage,
    registry
)

console = Console()

def register_custom_proportion_calculations():
    """Register custom proportion calculations with specific species."""
    
    # Example: Calculate proportion for Loblolly Pine (assuming it's at index 5)
    loblolly_proportion = SpeciesProportion(
        species_index=5, 
        species_code="SPCD0131"  # Loblolly Pine code
    )
    registry.register(loblolly_proportion)
    
    # Example: Calculate percentage for White Oak (assuming it's at index 3)
    white_oak_percentage = SpeciesPercentage(
        species_index=3,
        species_code="SPCD0012"  # White Oak code
    )
    registry.register(white_oak_percentage)
    
    # Example: Calculate proportion for Pine species group
    pine_group = SpeciesGroupProportion(
        species_indices=[5, 7, 12, 15],  # Indices of different pine species
        group_name="pines"
    )
    registry.register(pine_group)
    
    # Example: Calculate proportion for Oak species group
    oak_group = SpeciesGroupProportion(
        species_indices=[3, 8, 11, 14],  # Indices of different oak species
        group_name="oaks"
    )
    registry.register(oak_group)
    
    console.print("✅ Registered custom proportion calculations")

def create_species_proportion_config() -> BigMapSettings:
    """Create configuration for species proportion analysis."""
    
    return BigMapSettings(
        output_dir=Path("output/species_proportions"),
        calculations=[
            # Basic calculations
            CalculationConfig(
                name="total_biomass",
                enabled=True,
                output_format="geotiff"
            ),
            CalculationConfig(
                name="species_richness",
                enabled=True,
                parameters={"biomass_threshold": 1.0},
                output_format="geotiff"
            ),
            
            # Species proportion calculations
            CalculationConfig(
                name="species_proportion_SPCD0131",  # Loblolly Pine
                enabled=True,
                parameters={"species_index": 5},
                output_format="geotiff",
                output_name="loblolly_pine_proportion"
            ),
            CalculationConfig(
                name="species_percentage_SPCD0012",  # White Oak
                enabled=True,
                parameters={"species_index": 3},
                output_format="geotiff",
                output_name="white_oak_percentage"
            ),
            
            # Species group proportions
            CalculationConfig(
                name="species_group_proportion_pines",
                enabled=True,
                parameters={"species_indices": [5, 7, 12, 15]},
                output_format="geotiff",
                output_name="pine_group_proportion"
            ),
            CalculationConfig(
                name="species_group_proportion_oaks",
                enabled=True,
                parameters={"species_indices": [3, 8, 11, 14]},
                output_format="geotiff",
                output_name="oak_group_proportion"
            ),
        ]
    )

def analyze_species_proportions_from_zarr(zarr_path: str):
    """Analyze species proportions directly from zarr data."""
    
    console.print(f"\n[bold blue]🔍 Analyzing Species Proportions[/bold blue]")
    console.print(f"Zarr file: {zarr_path}")
    
    # Load zarr data
    zarr_array = zarr.open_array(zarr_path, mode='r')
    species_codes = zarr_array.attrs.get('species_codes', [])
    species_names = zarr_array.attrs.get('species_names', [])
    
    console.print(f"Shape: {zarr_array.shape}")
    console.print(f"Species count: {len(species_codes)}")
    
    # Calculate proportions for each species
    console.print(f"\n[bold green]📊 Species Proportion Summary[/bold green]")
    
    # Load a sample of data for analysis
    sample_data = zarr_array[:, :1000, :1000]  # Sample first 1000x1000 pixels
    
    # Calculate total biomass
    total_biomass = np.sum(sample_data, axis=0)
    forest_mask = total_biomass > 0
    forest_pixels = np.sum(forest_mask)
    
    if forest_pixels == 0:
        console.print("[red]No forest pixels found in sample area[/red]")
        return
    
    # Calculate proportion statistics for each species
    species_stats = []
    
    for i, (code, name) in enumerate(zip(species_codes, species_names)):
        species_biomass = sample_data[i]
        
        # Calculate proportions only for forest pixels
        if np.sum(species_biomass) > 0:
            proportions = np.zeros_like(total_biomass, dtype=np.float32)
            proportions[forest_mask] = species_biomass[forest_mask] / total_biomass[forest_mask]
            
            # Statistics
            mean_proportion = np.mean(proportions[forest_mask])
            max_proportion = np.max(proportions[forest_mask])
            pixels_present = np.sum(species_biomass > 0)
            coverage_pct = (pixels_present / forest_pixels) * 100
            
            species_stats.append({
                'index': i,
                'code': code,
                'name': name,
                'mean_proportion': mean_proportion,
                'max_proportion': max_proportion,
                'coverage_pct': coverage_pct,
                'pixels_present': pixels_present
            })
    
    # Sort by mean proportion (descending)
    species_stats.sort(key=lambda x: x['mean_proportion'], reverse=True)
    
    # Display results table
    table = Table(title="Species Proportion Analysis (Sample Area)")
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Species Code", style="cyan")
    table.add_column("Species Name", style="green", max_width=30)
    table.add_column("Mean %", justify="right", style="yellow")
    table.add_column("Max %", justify="right", style="red")
    table.add_column("Coverage %", justify="right", style="blue")
    table.add_column("Pixels", justify="right", style="magenta")
    
    for i, stats in enumerate(species_stats[:15], 1):  # Top 15 species
        table.add_row(
            str(i),
            stats['code'],
            stats['name'][:30],
            f"{stats['mean_proportion']*100:.2f}",
            f"{stats['max_proportion']*100:.1f}",
            f"{stats['coverage_pct']:.1f}",
            f"{stats['pixels_present']:,}"
        )
    
    console.print(table)
    
    # Show dominant species analysis
    console.print(f"\n[bold yellow]🏆 Dominance Analysis[/bold yellow]")
    
    top_5 = species_stats[:5]
    total_top5_proportion = sum(s['mean_proportion'] for s in top_5)
    
    console.print(f"Top 5 species account for {total_top5_proportion*100:.1f}% of total biomass")
    
    for i, stats in enumerate(top_5, 1):
        console.print(f"  {i}. {stats['name']}: {stats['mean_proportion']*100:.1f}%")

def run_proportion_analysis_pipeline(zarr_path: str):
    """Run the complete proportion analysis pipeline."""
    
    console.print("[bold blue]🌲 BigMap Species Proportion Analysis[/bold blue]")
    
    # Register custom calculations
    register_custom_proportion_calculations()
    
    # Create configuration
    settings = create_species_proportion_config()
    
    # Run analysis
    processor = ForestMetricsProcessor(settings)
    
    try:
        results = processor.run_analysis_pipeline(zarr_path)
        
        if results:
            console.print(f"\n[bold green]✅ Analysis Complete![/bold green]")
            console.print(f"Generated {len(results)} output files:")
            
            for name, path in results.items():
                console.print(f"  • {name} → {path}")
        else:
            console.print("[yellow]⚠️ No results generated[/yellow]")
            
    except Exception as e:
        console.print(f"[red]❌ Analysis failed: {e}[/red]")
        raise

def main():
    """Main function for species proportion analysis."""
    
    zarr_path = "output/nc_biomass_expandable.zarr"
    
    if not Path(zarr_path).exists():
        console.print(f"[red]❌ Zarr file not found: {zarr_path}[/red]")
        console.print("Please run the data preparation pipeline first.")
        return
    
    # Option 1: Quick analysis from zarr data
    console.print("[bold cyan]Option 1: Quick Proportion Analysis[/bold cyan]")
    analyze_species_proportions_from_zarr(zarr_path)
    
    # Option 2: Full pipeline with custom calculations
    console.print(f"\n[bold cyan]Option 2: Full Pipeline Analysis[/bold cyan]")
    
    # Ask user if they want to run the full pipeline
    response = input("\nRun full pipeline analysis? (y/N): ").lower().strip()
    if response in ['y', 'yes']:
        run_proportion_analysis_pipeline(zarr_path)
    else:
        console.print("Skipping full pipeline analysis.")
    
    console.print(f"\n[bold green]🎯 Next Steps:[/bold green]")
    console.print("1. Modify species indices in the script to match your data")
    console.print("2. Create custom species groups (e.g., hardwoods vs softwoods)")
    console.print("3. Use the generated GeoTIFF files for mapping and visualization")
    console.print("4. Combine with other ecological analyses")

if __name__ == "__main__":
    main() 