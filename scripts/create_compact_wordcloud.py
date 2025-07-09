#!/usr/bin/env python3
"""
Compact Species Word Cloud Generator

This script creates a highly clustered, compact word cloud of tree species
with maximum packing efficiency and minimal white space.

Author: BigMap Analysis Team
Date: 2025
"""

import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_and_process_data() -> Dict[str, float]:
    """Load and process species data to get biomass weights."""
    logger.info("Loading species data...")
    
    # Load existing weights from the previous analysis
    weights_df = pd.read_csv("output/species_wordcloud_weights.csv")
    weights = dict(zip(weights_df['species_name'], weights_df['total_biomass']))
    
    logger.info(f"Loaded {len(weights)} species weights")
    return weights

def create_ultra_compact_wordcloud(weights: Dict[str, float]) -> WordCloud:
    """Create an ultra-compact, tightly clustered word cloud."""
    logger.info("Generating ultra-compact word cloud...")
    
    # Ultra-compact configuration for maximum clustering
    wordcloud = WordCloud(
        width=800,              # Smaller, more square canvas
        height=600,             # Forces tighter packing
        max_words=80,           # Fewer words for better clustering
        relative_scaling=0.3,   # Much tighter scaling
        background_color='white',
        colormap='plasma',      # Alternative modern colormap
        collocations=False,
        prefer_horizontal=0.9,  # Mostly horizontal for efficient packing
        min_font_size=6,        # Very small minimum
        max_font_size=60,       # Moderate maximum
        font_path=None,
        random_state=42,
        scale=3,                # High scale for quality
        stopwords=set(),
        margin=2                # Minimal margins
    ).generate_from_frequencies(weights)
    
    logger.info("Ultra-compact word cloud generated")
    return wordcloud

def create_tight_layout_figure(wordcloud: WordCloud) -> plt.Figure:
    """Create a figure with minimal padding and maximum word density."""
    logger.info("Creating tight layout figure...")
    
    # Very compact figure setup
    fig, ax = plt.subplots(figsize=(10, 7.5), dpi=300, facecolor='white')
    
    # Display word cloud with no interpolation for crisp edges
    ax.imshow(wordcloud, interpolation='nearest')
    ax.axis('off')
    
    # Compact title
    fig.suptitle(
        'Forest Species: Heirs & Non-Heirs Properties\n(Biomass-Weighted Distribution)',
        fontsize=16,
        fontweight='bold',
        y=0.96,
        color='#2C3E50'
    )
    
    # Minimal subtitle
    fig.text(
        0.5, 0.01,
        'Word size ∝ Total biomass (Mg/ha) • 120 shared species',
        ha='center',
        fontsize=9,
        style='italic',
        color='#7F8C8D'
    )
    
    # Ultra-tight layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.02, right=0.98)
    
    return fig

def main():
    """Generate ultra-compact word cloud."""
    print("🌲 Ultra-Compact Species Word Cloud Generator")
    print("=" * 55)
    
    try:
        # Load data
        weights = load_and_process_data()
        
        # Generate compact word cloud
        wordcloud = create_ultra_compact_wordcloud(weights)
        
        # Create tight figure
        fig = create_tight_layout_figure(wordcloud)
        
        # Save compact version
        output_dir = Path("output")
        compact_path = output_dir / "species_wordcloud_compact.png"
        
        fig.savefig(
            compact_path,
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.1,  # Minimal padding
            facecolor='white',
            format='png'
        )
        
        logger.info(f"Saved compact word cloud: {compact_path}")
        
        # Display
        plt.show()
        
        print(f"\n✅ Compact word cloud saved: {compact_path}")
        print("This version maximizes word clustering and minimizes white space!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 