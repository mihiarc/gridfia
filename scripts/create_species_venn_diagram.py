#!/usr/bin/env python3
"""
Create Venn diagram comparing species composition between HEIRS and non-HEIRS properties.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib_venn as venn
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
import numpy as np

console = Console()

def load_species_data():
    """Load species summary data from both analyses."""
    
    # File paths
    heirs_file = Path("output/heirs_species_summary.csv")
    non_heirs_file = Path("output/non_heirs_species_summary.csv")
    
    # Check if files exist
    if not heirs_file.exists():
        console.print(f"[red]❌ HEIRS species file not found: {heirs_file}[/red]")
        console.print("Please run the HEIRS analysis first: python scripts/analyze_heirs_species_composition.py")
        return None, None
    
    if not non_heirs_file.exists():
        console.print(f"[red]❌ Non-HEIRS species file not found: {non_heirs_file}[/red]")
        console.print("Please run the non-HEIRS analysis first: python scripts/analyze_non_heirs_species.py")
        return None, None
    
    # Load data
    heirs_df = pd.read_csv(heirs_file, index_col=0)
    non_heirs_df = pd.read_csv(non_heirs_file, index_col=0)
    
    console.print(f"✅ Loaded HEIRS data: {len(heirs_df)} species")
    console.print(f"✅ Loaded non-HEIRS data: {len(non_heirs_df)} species")
    
    return heirs_df, non_heirs_df

def create_venn_diagram(heirs_df, non_heirs_df, output_dir="output"):
    """Create and save Venn diagram comparing species between property types."""
    
    # Get species codes (excluding TOTAL if present)
    heirs_species = set(heirs_df.index)
    non_heirs_species = set(non_heirs_df.index)
    
    # Remove TOTAL layer if present
    heirs_species.discard('TOTAL')
    non_heirs_species.discard('TOTAL')
    
    console.print(f"🔍 Analyzing species overlap:")
    console.print(f"  HEIRS properties: {len(heirs_species)} species")
    console.print(f"  Non-HEIRS properties: {len(non_heirs_species)} species")
    
    # Calculate overlaps
    overlap = heirs_species & non_heirs_species
    heirs_only = heirs_species - non_heirs_species
    non_heirs_only = non_heirs_species - heirs_species
    
    console.print(f"  Species found on both: {len(overlap)}")
    console.print(f"  HEIRS only: {len(heirs_only)}")
    console.print(f"  Non-HEIRS only: {len(non_heirs_only)}")
    
    # Create figure with larger size to accommodate species names
    plt.figure(figsize=(16, 12))
    
    # Create Venn diagram
    venn_diagram = venn.venn2(
        [heirs_species, non_heirs_species],
        set_labels=('HEIRS Properties', 'Non-HEIRS Properties'),
        set_colors=('lightblue', 'lightcoral'),
        alpha=0.7
    )
    
    # Customize the diagram
    plt.title('Species Composition Comparison:\nHEIRS vs Non-HEIRS Properties', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Create species name lists for display
    heirs_only_names = []
    non_heirs_only_names = []
    
    # Get species names for HEIRS-only species
    for species in heirs_only:
        if species in heirs_df.index:
            name = heirs_df.loc[species, 'name']
            # Shorten long names for display
            if len(name) > 25:
                name = name[:22] + "..."
            heirs_only_names.append(name)
    
    # Get species names for non-HEIRS-only species  
    for species in non_heirs_only:
        if species in non_heirs_df.index:
            name = non_heirs_df.loc[species, 'name']
            # Shorten long names for display
            if len(name) > 25:
                name = name[:22] + "..."
            non_heirs_only_names.append(name)
    
    # Add count labels with species names
    if venn_diagram.get_label_by_id('10'):  # HEIRS only
        heirs_text = f'{len(heirs_only)} species:\n' + '\n'.join(heirs_only_names)
        venn_diagram.get_label_by_id('10').set_text(heirs_text)
        venn_diagram.get_label_by_id('10').set_fontsize(9)
        venn_diagram.get_label_by_id('10').set_fontweight('bold')
    
    if venn_diagram.get_label_by_id('01'):  # Non-HEIRS only
        non_heirs_text = f'{len(non_heirs_only)} species:\n' + '\n'.join(non_heirs_only_names)
        venn_diagram.get_label_by_id('01').set_text(non_heirs_text)
        venn_diagram.get_label_by_id('01').set_fontsize(9)
        venn_diagram.get_label_by_id('01').set_fontweight('bold')
    
    if venn_diagram.get_label_by_id('11'):  # Overlap
        venn_diagram.get_label_by_id('11').set_text(f'{len(overlap)}\nspecies\n(shared)')
        venn_diagram.get_label_by_id('11').set_fontsize(12)
        venn_diagram.get_label_by_id('11').set_fontweight('bold')
    
    # Improve set labels
    if venn_diagram.get_label_by_id('A'):
        venn_diagram.get_label_by_id('A').set_fontsize(14)
        venn_diagram.get_label_by_id('A').set_fontweight('bold')
    
    if venn_diagram.get_label_by_id('B'):
        venn_diagram.get_label_by_id('B').set_fontsize(14)
        venn_diagram.get_label_by_id('B').set_fontweight('bold')
    
    # Add summary statistics as text box
    total_heirs = len(heirs_species)
    total_non_heirs = len(non_heirs_species)
    total_unique = len(heirs_species | non_heirs_species)
    
    overlap_pct_heirs = (len(overlap) / total_heirs * 100) if total_heirs > 0 else 0
    overlap_pct_non_heirs = (len(overlap) / total_non_heirs * 100) if total_non_heirs > 0 else 0
    
    stats_text = f"""Summary Statistics:
    
Total Unique Species: {total_unique}
Shared Species: {len(overlap)} ({overlap_pct_heirs:.1f}% of HEIRS, {overlap_pct_non_heirs:.1f}% of Non-HEIRS)

HEIRS Properties:
• Total species: {total_heirs}
• Exclusive species: {len(heirs_only)} ({len(heirs_only)/total_heirs*100:.1f}%)

Non-HEIRS Properties:
• Total species: {total_non_heirs} 
• Exclusive species: {len(non_heirs_only)} ({len(non_heirs_only)/total_non_heirs*100:.1f}%)"""
    
    plt.text(1.2, 0.5, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    
    # Save the diagram
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    venn_file = output_path / "species_venn_diagram.png"
    plt.savefig(venn_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_path / "species_venn_diagram.pdf", bbox_inches='tight')
    
    console.print(f"💾 Saved Venn diagram: {venn_file}")
    console.print(f"💾 Saved PDF version: {output_path / 'species_venn_diagram.pdf'}")
    
    # Show the plot
    plt.show()
    
    return {
        'heirs_species': heirs_species,
        'non_heirs_species': non_heirs_species,
        'overlap': overlap,
        'heirs_only': heirs_only,
        'non_heirs_only': non_heirs_only
    }

def analyze_species_differences(heirs_df, non_heirs_df, venn_data):
    """Analyze characteristics of species unique to each property type."""
    
    console.print(f"\n[bold cyan]📊 Species Difference Analysis[/bold cyan]")
    
    heirs_only = venn_data['heirs_only']
    non_heirs_only = venn_data['non_heirs_only']
    
    # Analyze HEIRS-only species
    if heirs_only:
        console.print(f"\n[bold blue]🏠 Species Found ONLY on HEIRS Properties ({len(heirs_only)} species):[/bold blue]")
        heirs_unique_df = heirs_df.loc[list(heirs_only)].sort_values('occurrence_rate', ascending=False)
        
        for species in heirs_unique_df.head(10).index:
            data = heirs_unique_df.loc[species]
            console.print(f"  • {species}: {data['name']} - {data['occurrence_rate']*100:.1f}% occurrence")
    
    # Analyze non-HEIRS-only species
    if non_heirs_only:
        console.print(f"\n[bold red]🏢 Species Found ONLY on Non-HEIRS Properties ({len(non_heirs_only)} species):[/bold red]")
        non_heirs_unique_df = non_heirs_df.loc[list(non_heirs_only)].sort_values('occurrence_rate', ascending=False)
        
        for species in non_heirs_unique_df.head(10).index:
            data = non_heirs_unique_df.loc[species]
            console.print(f"  • {species}: {data['name']} - {data['occurrence_rate']*100:.1f}% occurrence")
    
    # Save detailed comparison
    output_path = Path("output")
    
    # Species comparison summary
    comparison_data = []
    
    # Add shared species
    for species in venn_data['overlap']:
        heirs_data = heirs_df.loc[species] if species in heirs_df.index else None
        non_heirs_data = non_heirs_df.loc[species] if species in non_heirs_df.index else None
        
        comparison_data.append({
            'species_code': species,
            'species_name': heirs_data['name'] if heirs_data is not None else non_heirs_data['name'],
            'property_type': 'Both',
            'heirs_occurrence': heirs_data['occurrence_rate'] if heirs_data is not None else 0,
            'non_heirs_occurrence': non_heirs_data['occurrence_rate'] if non_heirs_data is not None else 0,
            'heirs_parcels': heirs_data['parcel_count'] if heirs_data is not None else 0,
            'non_heirs_parcels': non_heirs_data['parcel_count'] if non_heirs_data is not None else 0
        })
    
    # Add HEIRS-only species
    for species in heirs_only:
        heirs_data = heirs_df.loc[species]
        comparison_data.append({
            'species_code': species,
            'species_name': heirs_data['name'],
            'property_type': 'HEIRS Only',
            'heirs_occurrence': heirs_data['occurrence_rate'],
            'non_heirs_occurrence': 0,
            'heirs_parcels': heirs_data['parcel_count'],
            'non_heirs_parcels': 0
        })
    
    # Add non-HEIRS-only species
    for species in non_heirs_only:
        non_heirs_data = non_heirs_df.loc[species]
        comparison_data.append({
            'species_code': species,
            'species_name': non_heirs_data['name'],
            'property_type': 'Non-HEIRS Only',
            'heirs_occurrence': 0,
            'non_heirs_occurrence': non_heirs_data['occurrence_rate'],
            'heirs_parcels': 0,
            'non_heirs_parcels': non_heirs_data['parcel_count']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_file = output_path / "species_comparison_analysis.csv"
    comparison_df.to_csv(comparison_file, index=False)
    console.print(f"💾 Saved detailed comparison: {comparison_file}")

def main():
    """Main function to create species comparison Venn diagram."""
    
    # Header
    header = Panel(
        "[bold]Species Composition Venn Diagram Analysis[/bold]\n\n"
        "Comparing species found on HEIRS vs Non-HEIRS properties to identify:\n"
        "• Species shared between both property types\n"
        "• Species unique to HEIRS properties\n"
        "• Species unique to Non-HEIRS properties",
        title="🌲 Species Overlap Analysis",
        border_style="magenta"
    )
    console.print(header)
    
    try:
        # Load species data
        heirs_df, non_heirs_df = load_species_data()
        
        if heirs_df is None or non_heirs_df is None:
            return
        
        # Create Venn diagram
        venn_data = create_venn_diagram(heirs_df, non_heirs_df)
        
        # Analyze differences
        analyze_species_differences(heirs_df, non_heirs_df, venn_data)
        
        console.print(f"\n[bold green]✅ Analysis Complete![/bold green]")
        console.print("Generated files:")
        console.print("• species_venn_diagram.png - Visual comparison")
        console.print("• species_venn_diagram.pdf - PDF version")
        console.print("• species_comparison_analysis.csv - Detailed data")
        
    except Exception as e:
        console.print(f"[red]❌ Error creating Venn diagram: {str(e)}[/red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 