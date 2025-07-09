#!/usr/bin/env python3
"""
Analyze species composition at NON-HEIRS property locations.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from analyze_heirs_species_composition import HeirsSpeciesAnalyzer
from rich.console import Console
from rich.panel import Panel
import pandas as pd

console = Console()

def main():
    """Main analysis function for non-heirs properties."""
    
    # Configuration for non-heirs analysis
    parquet_path = "data/non_heirs_with_species_diversity.parquet"
    zarr_path = "output/nc_biomass_expandable.zarr"
    biomass_threshold = 0.1  # Mg/ha
    
    # Header
    header = Panel(
        "[bold]NON-HEIRS Property Species Composition Analysis[/bold]\n\n"
        "Analyzing species composition at NON-HEIRS property locations using:\n"
        f"• Parcel data: {parquet_path}\n"
        f"• Species zarr: {zarr_path}\n"
        f"• Biomass threshold: {biomass_threshold} Mg/ha",
        title="🌲 BigMap NON-HEIRS Analysis",
        border_style="green"
    )
    console.print(header)
    
    try:
        # Initialize analyzer
        analyzer = HeirsSpeciesAnalyzer(zarr_path)
        
        # Load non-heirs parcel data
        parcels = analyzer.load_parcel_data(parquet_path)
        
        # Extract species data
        results = analyzer.extract_species_at_parcels(parcels, biomass_threshold)
        
        # Analyze composition
        analysis = analyzer.analyze_species_composition(results)
        
        # Display results
        analyzer.display_results(analysis)
        
        # Save results with different names
        output_path = Path("output")
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save non-heirs results
        results.to_csv(output_path / "non_heirs_parcel_species_composition.csv", index=False)
        console.print(f"💾 Saved non-heirs parcel results: {output_path / 'non_heirs_parcel_species_composition.csv'}")
        
        if analysis.get('species_details'):
            species_df = pd.DataFrame.from_dict(analysis['species_details'], orient='index')
            species_df.to_csv(output_path / "non_heirs_species_summary.csv")
            console.print(f"💾 Saved non-heirs species summary: {output_path / 'non_heirs_species_summary.csv'}")
        
        # Save analysis metadata
        import json
        metadata_output = output_path / "non_heirs_analysis_metadata.json"
        with open(metadata_output, 'w') as f:
            clean_analysis = {
                'total_parcels_analyzed': int(analysis['total_parcels_analyzed']),
                'parcels_with_biomass': int(analysis['parcels_with_biomass']),
                'parcels_outside_coverage': int(analysis['parcels_outside_coverage']),
                'unique_species_found': int(analysis['unique_species_found'])
            }
            json.dump(clean_analysis, f, indent=2)
        console.print(f"💾 Saved non-heirs metadata: {metadata_output}")
        
        console.print(f"\n[bold cyan]🎯 Comparison Ready:[/bold cyan]")
        console.print("• HEIRS results: output/heirs_*")
        console.print("• NON-HEIRS results: output/non_heirs_*")
        console.print("• Ready for comparative analysis!")
        
        return results, analysis
        
    except Exception as e:
        console.print(f"[red]❌ Analysis failed: {str(e)}[/red]")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    main() 