#!/usr/bin/env python3
"""
Hybrid Venn Diagram + Word Cloud Visualization

This script creates an innovative visualization combining a Venn diagram
showing species overlap with a word cloud of the top shared species
positioned in the center overlap area.

Author: BigMap Analysis Team
Date: 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib_venn as venn
from wordcloud import WordCloud
from pathlib import Path
import numpy as np
import logging
from typing import Dict, Tuple, Set
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class VennWordCloudHybrid:
    """Create hybrid Venn diagram with embedded word cloud."""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def load_species_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load species data from all three sources."""
        logger.info("Loading species data...")
        
        # Load data files
        heirs_df = pd.read_csv("output/heirs_species_summary.csv", index_col=0)
        non_heirs_df = pd.read_csv("output/non_heirs_species_summary.csv", index_col=0)
        comparison_df = pd.read_csv("output/species_comparison_analysis.csv")
        
        # Remove TOTAL if present
        heirs_df = heirs_df.drop('TOTAL', errors='ignore')
        non_heirs_df = non_heirs_df.drop('TOTAL', errors='ignore')
        
        logger.info(f"Loaded {len(heirs_df)} HEIRS species, {len(non_heirs_df)} non-HEIRS species")
        
        return heirs_df, non_heirs_df, comparison_df
    
    def get_shared_species_weights(self, comparison_df: pd.DataFrame) -> Dict[str, float]:
        """Get biomass weights for shared species only."""
        logger.info("Extracting shared species weights...")
        
        # Filter for shared species
        shared_species = comparison_df[comparison_df['property_type'] == 'Both'].copy()
        
        # Load existing biomass weights
        weights_df = pd.read_csv("output/species_wordcloud_weights.csv")
        weights_dict = dict(zip(weights_df['species_name'], weights_df['total_biomass']))
        
        # Filter to only shared species and get top 20
        shared_weights = {}
        for _, row in shared_species.iterrows():
            species_name = row['species_name']
            if species_name in weights_dict:
                shared_weights[species_name] = weights_dict[species_name]
        
        # Use ALL shared species by biomass
        all_shared_weights = shared_weights
        
        logger.info(f"Selected ALL {len(all_shared_weights)} shared species by biomass")
        logger.info(f"Top 5: {list(sorted(all_shared_weights.items(), key=lambda x: x[1], reverse=True))[:5]}")
        
        return all_shared_weights
    
    def create_circular_mask(self, size: int = 400) -> np.ndarray:
        """Create a perfect circular mask for the word cloud that matches Venn diagram circles."""
        # Create a square array filled with black (excluded area)
        mask = np.zeros((size, size), dtype=np.uint8)
        
        # Create a perfect circle in the center
        center = size // 2
        radius = center - 10  # Small padding from edge
        
        # Use meshgrid to create coordinate arrays
        y, x = np.ogrid[:size, :size]
        
        # Create perfect circular mask
        # White (255) = included in wordcloud, Black (0) = excluded
        circle_mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
        mask[circle_mask] = 255
        
        return mask

    def create_compact_wordcloud_image(self, weights: Dict[str, float]) -> np.ndarray:
        """Create a compact square word cloud for Venn diagram overlay."""
        logger.info("Creating compact square word cloud for overlay...")
        
        # Create square word cloud for ALL shared species
        wordcloud = WordCloud(
            width=500,              # Square dimensions for better fit
            height=400,
            max_words=200,          # Allow all species (safety buffer)
            relative_scaling=0.2,   # Tight scaling for maximum density
            background_color=None,  # Transparent background
            mode='RGBA',            # Include alpha channel
            colormap='viridis',
            collocations=False,
            prefer_horizontal=0.7,  # Mix orientations for dense packing
            min_font_size=4,        # Small minimum for more species
            max_font_size=28,       # Moderate max to fit more words
            font_path=None,
            random_state=42,
            margin=1,               # Minimal margin for tight packing
            scale=2
        ).generate_from_frequencies(weights)
        
        # Convert to numpy array
        img_array = np.array(wordcloud.to_image())
        
        logger.info("Compact square word cloud created")
        return img_array
    
    def create_venn_sets(self, heirs_df: pd.DataFrame, non_heirs_df: pd.DataFrame) -> Dict[str, Set]:
        """Create sets for Venn diagram."""
        heirs_species = set(heirs_df.index)
        non_heirs_species = set(non_heirs_df.index)
        
        overlap = heirs_species & non_heirs_species
        heirs_only = heirs_species - non_heirs_species
        non_heirs_only = non_heirs_species - heirs_species
        
        logger.info(f"Venn diagram sets: {len(heirs_only)} HEIRS-only, {len(overlap)} shared, {len(non_heirs_only)} non-HEIRS-only")
        
        return {
            'heirs_species': heirs_species,
            'non_heirs_species': non_heirs_species,
            'overlap': overlap,
            'heirs_only': heirs_only,
            'non_heirs_only': non_heirs_only
        }
    
    def create_hybrid_visualization(self, venn_sets: Dict, wordcloud_img: np.ndarray, 
                                  heirs_df: pd.DataFrame, non_heirs_df: pd.DataFrame) -> plt.Figure:
        """Create the hybrid Venn diagram with word cloud overlay."""
        logger.info("Creating hybrid visualization...")
        
        # Create figure with optimal proportions
        fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
        
        # Create Venn diagram
        venn_diagram = venn.venn2(
            [venn_sets['heirs_species'], venn_sets['non_heirs_species']],
            set_labels=('HEIRS Properties', 'Non-HEIRS Properties'),
            set_colors=('lightblue', 'lightcoral'),
            alpha=0.6,  # Slightly more transparent to show word cloud better
            ax=ax
        )
        
        # Get species names for exclusive areas
        heirs_only_names = self.get_species_names(venn_sets['heirs_only'], heirs_df, max_names=4)
        non_heirs_only_names = self.get_species_names(venn_sets['non_heirs_only'], non_heirs_df, max_names=5)
        
        # Update labels with clean formatting (no redundant count labels)
        if venn_diagram.get_label_by_id('10'):  # HEIRS only
            heirs_text = '\n'.join(heirs_only_names)
            venn_diagram.get_label_by_id('10').set_text(heirs_text)
            venn_diagram.get_label_by_id('10').set_fontsize(10)
            venn_diagram.get_label_by_id('10').set_fontweight('normal')
        
        if venn_diagram.get_label_by_id('01'):  # Non-HEIRS only
            non_heirs_text = '\n'.join(non_heirs_only_names)
            venn_diagram.get_label_by_id('01').set_text(non_heirs_text)
            venn_diagram.get_label_by_id('01').set_fontsize(10)
            venn_diagram.get_label_by_id('01').set_fontweight('normal')
        
        # Clear the center label - we'll replace it with word cloud
        if venn_diagram.get_label_by_id('11'):
            venn_diagram.get_label_by_id('11').set_text('')
        
        # Add compact square word cloud to center overlap
        center_x, center_y = 0, 0  # Center of Venn overlap
        
        # Create OffsetImage for the word cloud with appropriate zoom
        imagebox = OffsetImage(wordcloud_img, zoom=0.4)  # Optimal zoom for square fit
        
        # Position the word cloud in the center
        ab = AnnotationBbox(imagebox, (center_x, center_y), frameon=False, 
                          xycoords='data', boxcoords="offset points")
        ax.add_artist(ab)
        

        
        # No title for minimalist visualization
        
        # No caption for minimalist visualization
        plt.tight_layout()
        
        return fig
    
    def _add_external_species_lists(self, ax, heirs_only_names: list, non_heirs_only_names: list, 
                                   heirs_count: int, non_heirs_count: int) -> None:
        """Add exclusive species lists outside the Venn diagram."""
        
        # HEIRS-only species (left side)
        heirs_title = f"HEIRS Exclusive ({heirs_count} species):"
        heirs_species_text = heirs_title + '\n' + '\n'.join(f"• {name}" for name in heirs_only_names)
        ax.text(-1.3, 0.5, heirs_species_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='top', horizontalalignment='left',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.3, edgecolor="gray"))
        
        # Non-HEIRS-only species (right side)  
        non_heirs_title = f"Non-HEIRS Exclusive ({non_heirs_count} species):"
        non_heirs_species_text = non_heirs_title + '\n' + '\n'.join(f"• {name}" for name in non_heirs_only_names)
        ax.text(1.3, 0.5, non_heirs_species_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.3, edgecolor="gray"))

    def get_species_names(self, species_codes: Set, df: pd.DataFrame, max_names: int = 5) -> list:
        """Get readable species names for a set of species codes."""
        names = []
        for code in list(species_codes)[:max_names]:
            if code in df.index:
                name = df.loc[code, 'name']
                # Shorten long names
                if len(name) > 20:
                    name = name[:17] + "..."
                names.append(name)
        return names
    
    def save_outputs(self, fig: plt.Figure) -> None:
        """Save the hybrid visualization."""
        logger.info("Saving hybrid visualization...")
        
        # Save high-resolution version
        output_path = self.output_dir / "species_venn_wordcloud_hybrid.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved hybrid visualization: {output_path}")
        
        # Save PDF version
        pdf_path = self.output_dir / "species_venn_wordcloud_hybrid.pdf"
        fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved PDF version: {pdf_path}")
    
    def run_analysis(self) -> None:
        """Execute the complete hybrid visualization pipeline."""
        logger.info("Starting hybrid Venn diagram + word cloud creation...")
        
        try:
            # Load data
            heirs_df, non_heirs_df, comparison_df = self.load_species_data()
            
            # Get shared species weights (ALL shared species)
            shared_weights = self.get_shared_species_weights(comparison_df)
            
            # Create compact word cloud image
            wordcloud_img = self.create_compact_wordcloud_image(shared_weights)
            
            # Create Venn sets
            venn_sets = self.create_venn_sets(heirs_df, non_heirs_df)
            
            # Create hybrid visualization
            fig = self.create_hybrid_visualization(venn_sets, wordcloud_img, heirs_df, non_heirs_df)
            
            # Save outputs
            self.save_outputs(fig)
            
            # Display
            plt.show()
            
            logger.info("Hybrid visualization completed successfully!")
            
        except Exception as e:
            logger.error(f"Error creating hybrid visualization: {e}")
            raise


def main():
    """Main execution function."""
    print("🌲🔄 Hybrid Venn Diagram + Word Cloud Generator")
    print("=" * 60)
    print("Creating innovative visualization combining:")
    print("  📊 Venn diagram showing species overlap")
    print("  ☁️  Word cloud of ALL shared species (biomass-weighted)")
    print("=" * 60)
    
    try:
        generator = VennWordCloudHybrid()
        generator.run_analysis()
        
        print("\n✅ Hybrid visualization completed!")
        print("Check the 'output' directory for results:")
        print("  🎨 species_venn_wordcloud_hybrid.png - Innovative hybrid visualization")
        print("  📄 species_venn_wordcloud_hybrid.pdf - PDF version")
        print("\nThis combines the categorical power of Venn diagrams")
        print("with the quantitative insight of biomass-weighted word clouds!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 