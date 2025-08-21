#!/usr/bin/env python
"""
Run biomass analysis for all isochrone time intervals.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analyze_mill_90min_biomass import analyze_mill_isochrone_biomass
from rich.console import Console
import pandas as pd

console = Console()

def run_all_analyses():
    """Run biomass analysis for all available isochrones."""
    
    console.print("[bold cyan]Running Biomass Analysis for All Isochrone Time Intervals[/bold cyan]\n")
    
    # Define time intervals to analyze
    time_intervals = [30, 45, 60, 90, 120]
    
    # Store results for comparison
    all_results = []
    
    for time_min in time_intervals:
        console.print(f"\n[yellow]{'='*60}[/yellow]")
        console.print(f"[bold]Analyzing {time_min}-minute isochrone...[/bold]")
        console.print(f"[yellow]{'='*60}[/yellow]\n")
        
        try:
            results = analyze_mill_isochrone_biomass(time_minutes=time_min)
            
            if results:
                # Extract summary statistics
                timber_result = results['results'][results['results']['code'] == 'TMBR'].iloc[0]
                
                summary = {
                    'time_minutes': time_min,
                    'total_area_ha': results['mask'].sum() * (98.425197**2) * 0.0000092903,  # pixels to hectares
                    'forest_area_ha': timber_result['forest_area_ha'],
                    'forest_coverage_pct': timber_result['coverage_pct'],
                    'total_biomass_mg': timber_result['total_biomass_mg'],
                    'avg_density_mg_ha': timber_result['mean_biomass_mg_ha']
                }
                
                all_results.append(summary)
                console.print(f"[green]✓ Completed {time_min}-minute analysis[/green]")
            else:
                console.print(f"[red]✗ Failed {time_min}-minute analysis[/red]")
                
        except Exception as e:
            console.print(f"[red]Error analyzing {time_min}-minute isochrone: {e}[/red]")
            continue
    
    # Create summary DataFrame
    if all_results:
        summary_df = pd.DataFrame(all_results)
        
        # Display summary table
        console.print("\n[bold cyan]Summary of All Analyses:[/bold cyan]\n")
        console.print(summary_df.to_string(index=False))
        
        # Save summary to CSV
        summary_path = Path("output/analysis/isochrone_biomass_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        console.print(f"\n[green]✓ Summary saved to: {summary_path}[/green]")
        
        return summary_df
    
    return None


if __name__ == "__main__":
    run_all_analyses()