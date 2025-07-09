#!/usr/bin/env python3
"""
Southern Yellow Pine Analysis for North Carolina

This script analyzes the distribution and proportion of Southern Yellow Pine
species (Loblolly, Shortleaf, Longleaf, and Slash Pine) across North Carolina.
"""

from pathlib import Path
import numpy as np
import zarr
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from bigmap.config import BigMapSettings, CalculationConfig
from bigmap.core.forest_metrics import ForestMetricsProcessor
from bigmap.core.calculations import SpeciesGroupProportion, registry

console = Console()

# Southern Yellow Pine species configuration
SYP_SPECIES = {
    39: {"code": "SPCD0110", "name": "Shortleaf Pine", "scientific": "Pinus echinata"},
    40: {"code": "SPCD0111", "name": "Slash Pine", "scientific": "Pinus elliottii"},
    44: {"code": "SPCD0121", "name": "Longleaf Pine", "scientific": "Pinus palustris"},
    50: {"code": "SPCD0131", "name": "Loblolly Pine", "scientific": "Pinus taeda"}
}

SYP_INDICES = list(SYP_SPECIES.keys())

def register_syp_calculations():
    """Register Southern Yellow Pine specific calculations."""
    
    # Register the group calculation
    syp_group = SpeciesGroupProportion(
        species_indices=SYP_INDICES,
        group_name="southern_yellow_pine",
        exclude_total_layer=True
    )
    registry.register(syp_group)
    
    console.print("✅ Registered Southern Yellow Pine calculations")

def analyze_syp_from_zarr(zarr_path: str):
    """Analyze Southern Yellow Pine species directly from zarr data."""
    
    console.print(f"\n[bold blue]🌲 Southern Yellow Pine Analysis[/bold blue]")
    console.print(f"Dataset: {zarr_path}")
    
    # Load zarr data
    zarr_array = zarr.open_array(zarr_path, mode='r')
    species_codes = zarr_array.attrs.get('species_codes', [])
    species_names = zarr_array.attrs.get('species_names', [])
    
    console.print(f"Total species: {len(species_codes)}")
    
    # Verify our species are present
    console.print(f"\n[bold green]🔍 Species Verification[/bold green]")
    verification_table = Table(title="Southern Yellow Pine Species in Dataset")
    verification_table.add_column("Index", style="cyan", width=6)
    verification_table.add_column("Code", style="yellow", width=10)
    verification_table.add_column("Common Name", style="green")
    verification_table.add_column("Scientific Name", style="blue")
    verification_table.add_column("Status", style="magenta")
    
    verified_species = []
    for index in SYP_INDICES:
        if index < len(species_codes):
            actual_code = species_codes[index]
            actual_name = species_names[index]
            expected = SYP_SPECIES[index]
            
            # Check if codes match
            status = "✅ Found" if expected["code"] in actual_code else "⚠️ Code mismatch"
            
            verification_table.add_row(
                str(index),
                actual_code,
                actual_name,
                expected["scientific"],
                status
            )
            
            if "✅" in status:
                verified_species.append(index)
        else:
            verification_table.add_row(
                str(index),
                "N/A",
                "Index out of range",
                SYP_SPECIES[index]["scientific"],
                "❌ Missing"
            )
    
    console.print(verification_table)
    
    if not verified_species:
        console.print("[red]❌ No Southern Yellow Pine species found![/red]")
        return
    
    # Sample analysis (first 2000x2000 pixels for speed)
    console.print(f"\n[bold green]📊 Biomass Analysis[/bold green] (Sample: 2000x2000 pixels)")
    
    sample_data = zarr_array[:, :2000, :2000]
    
    # Calculate total biomass from individual species (excluding index 0)
    total_biomass = np.sum(sample_data[1:], axis=0)
    forest_mask = total_biomass > 1.0  # Areas with >1 Mg/ha biomass
    forest_pixels = np.sum(forest_mask)
    
    if forest_pixels == 0:
        console.print("[red]No significant forest biomass found in sample area[/red]")
        return
    
    # Analyze each SYP species
    syp_stats = []
    total_syp_biomass = np.zeros_like(total_biomass)
    
    for index in verified_species:
        species_data = sample_data[index]
        species_info = SYP_SPECIES[index]
        
        # Add to total SYP biomass
        total_syp_biomass += species_data
        
        # Calculate statistics
        presence_mask = species_data > 0.5  # Presence threshold
        presence_pixels = np.sum(presence_mask & forest_mask)
        coverage_pct = (presence_pixels / forest_pixels) * 100 if forest_pixels > 0 else 0
        
        # Mean biomass where present
        mean_biomass = np.mean(species_data[presence_mask & forest_mask]) if presence_pixels > 0 else 0
        max_biomass = np.max(species_data) if presence_pixels > 0 else 0
        
        # Proportion of total forest biomass
        proportion = np.sum(species_data[forest_mask]) / np.sum(total_biomass[forest_mask]) * 100 if forest_pixels > 0 else 0
        
        syp_stats.append({
            'index': index,
            'name': species_info['name'],
            'coverage_pct': coverage_pct,
            'presence_pixels': presence_pixels,
            'mean_biomass': mean_biomass,
            'max_biomass': max_biomass,
            'proportion_of_total': proportion
        })
    
    # Sort by proportion of total biomass
    syp_stats.sort(key=lambda x: x['proportion_of_total'], reverse=True)
    
    # Display results
    results_table = Table(title="Southern Yellow Pine Species Analysis")
    results_table.add_column("Species", style="green", max_width=20)
    results_table.add_column("Coverage %", justify="right", style="cyan")
    results_table.add_column("Pixels", justify="right", style="blue")
    results_table.add_column("Mean Biomass", justify="right", style="yellow")
    results_table.add_column("Max Biomass", justify="right", style="red")
    results_table.add_column("% of Total", justify="right", style="magenta")
    
    total_syp_proportion = 0
    for stats in syp_stats:
        results_table.add_row(
            stats['name'],
            f"{stats['coverage_pct']:.1f}",
            f"{stats['presence_pixels']:,}",
            f"{stats['mean_biomass']:.1f}",
            f"{stats['max_biomass']:.1f}",
            f"{stats['proportion_of_total']:.2f}"
        )
        total_syp_proportion += stats['proportion_of_total']
    
    console.print(results_table)
    
    # Summary statistics
    syp_presence_mask = total_syp_biomass > 0.5
    syp_coverage = np.sum(syp_presence_mask & forest_mask) / forest_pixels * 100 if forest_pixels > 0 else 0
    
    summary_panel = Panel(
        f"[bold]Southern Yellow Pine Group Summary[/bold]\n\n"
        f"🌲 Total SYP Proportion: [yellow]{total_syp_proportion:.1f}%[/yellow] of forest biomass\n"
        f"📍 Coverage Area: [cyan]{syp_coverage:.1f}%[/cyan] of forest pixels\n"
        f"🏆 Dominant Species: [green]{syp_stats[0]['name']}[/green] ({syp_stats[0]['proportion_of_total']:.1f}%)\n"
        f"🔢 Total Forest Pixels: [blue]{forest_pixels:,}[/blue]\n"
        f"📊 Sample Area: [dim]2000×2000 pixels[/dim]",
        title="📊 Analysis Summary",
        border_style="green"
    )
    
    console.print(f"\n{summary_panel}")

def run_syp_pipeline(zarr_path: str):
    """Run the complete Southern Yellow Pine analysis pipeline."""
    
    console.print("[bold blue]🚀 Running Southern Yellow Pine Pipeline[/bold blue]")
    
    # Register custom calculations
    register_syp_calculations()
    
    # Create configuration
    settings = BigMapSettings(
        output_dir=Path("output/southern_yellow_pine_analysis"),
        calculations=[
            CalculationConfig(
                name="total_biomass",
                enabled=True,
                parameters={"exclude_total_layer": True},
                output_format="geotiff",
                output_name="nc_total_biomass"
            ),
            CalculationConfig(
                name="species_group_proportion_southern_yellow_pine",
                enabled=True,
                parameters={
                    "species_indices": SYP_INDICES,
                    "group_name": "southern_yellow_pine",
                    "exclude_total_layer": True
                },
                output_format="geotiff",
                output_name="southern_yellow_pine_proportion"
            )
        ]
    )
    
    # Run analysis
    processor = ForestMetricsProcessor(settings)
    
    try:
        results = processor.run_analysis_pipeline(zarr_path)
        
        if results:
            console.print(f"\n[bold green]✅ Pipeline Complete![/bold green]")
            console.print(f"Generated {len(results)} output files:")
            
            for name, path in results.items():
                console.print(f"  • [cyan]{name}[/cyan] → {path}")
        else:
            console.print("[yellow]⚠️ No results generated[/yellow]")
            
    except Exception as e:
        console.print(f"[red]❌ Pipeline failed: {e}[/red]")
        raise

def main():
    """Main function for Southern Yellow Pine analysis."""
    
    zarr_path = "output/nc_biomass_expandable.zarr"
    
    if not Path(zarr_path).exists():
        console.print(f"[red]❌ Zarr file not found: {zarr_path}[/red]")
        console.print("Please ensure the biomass data is available.")
        return
    
    # Header
    header = Panel(
        "[bold]Southern Yellow Pine Analysis for North Carolina[/bold]\n\n"
        "Analyzing the distribution and biomass proportion of:\n"
        "• Loblolly Pine (Pinus taeda)\n"
        "• Shortleaf Pine (Pinus echinata)\n"
        "• Longleaf Pine (Pinus palustris)\n"
        "• Slash Pine (Pinus elliottii)",
        title="🌲 BigMap SYP Analysis",
        border_style="green"
    )
    console.print(header)
    
    # Quick analysis
    analyze_syp_from_zarr(zarr_path)
    
    # Ask about full pipeline
    console.print(f"\n[bold cyan]Full Pipeline Options:[/bold cyan]")
    console.print("1. Quick analysis (completed above)")
    console.print("2. Full pipeline with GeoTIFF outputs")
    console.print("3. Use pre-configured YAML file")
    
    choice = input("\nSelect option (1/2/3) or press Enter to exit: ").strip()
    
    if choice == "2":
        run_syp_pipeline(zarr_path)
    elif choice == "3":
        console.print(f"\n[bold yellow]To run with configuration file:[/bold yellow]")
        console.print("bigmap calculate output/nc_biomass_expandable.zarr --config cfg/species/southern_yellow_pine_config.yaml")
    
    console.print(f"\n[bold green]🎯 Next Steps:[/bold green]")
    console.print("• Use the generated maps for spatial analysis")
    console.print("• Compare SYP distribution with environmental variables")
    console.print("• Analyze temporal changes if historical data available")
    console.print("• Focus on Longleaf Pine for conservation planning")

if __name__ == "__main__":
    main() 