"""
Vance County NDVI Visualization Module

This module handles the creation of visualizations for NDVI analysis results,
including trend plots, distributions, and comparative visualizations between
heirs and non-heirs properties.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Optional, Tuple, Dict

from ..config.analysis_config import (
    ANALYSIS_YEARS,
    FIGURE_SIZE,
    PLOT_DPI,
    COLOR_SCHEME,
    LINE_STYLES,
    LOG_FORMAT
)

# Configure logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class NDVIPlotter:
    """Handles visualization of NDVI analysis results."""
    
    def __init__(self, ndvi_data: Optional[pd.DataFrame] = None):
        """Initialize the NDVI plotter.
        
        Args:
            ndvi_data: Optional DataFrame containing NDVI data
        """
        self.data = ndvi_data
        self._setup_style()
        
    def _setup_style(self):
        """Configure plot style settings."""
        plt.style.use('seaborn')
        sns.set_palette('deep')
        
    def load_data(self, ndvi_file: Path) -> pd.DataFrame:
        """Load NDVI data from processed parquet file.
        
        Args:
            ndvi_file: Path to NDVI data file
            
        Returns:
            DataFrame containing NDVI data
        """
        try:
            df = pd.read_parquet(ndvi_file)
            logger.info(f"Loaded NDVI data with {len(df)} records")
            self.data = df
            return df
        except Exception as e:
            logger.error(f"Error loading NDVI data: {e}")
            raise
            
    def plot_trend_comparison(self, save_path: Optional[Path] = None) -> None:
        """Create visualization comparing NDVI trends between groups.
        
        Args:
            save_path: Optional path to save the plot
        """
        if self.data is None:
            raise ValueError("No data loaded")
            
        plt.figure(figsize=FIGURE_SIZE)
        
        # Trend slope distribution
        plt.subplot(1, 2, 1)
        sns.boxplot(
            data=self.data,
            x='is_heir',
            y='ndvi_trend_slope',
            palette=COLOR_SCHEME
        )
        plt.title('NDVI Trend Slope Distribution')
        plt.xlabel('Heirs Property')
        plt.ylabel('Trend Slope')
        
        # NDVI values over time
        plt.subplot(1, 2, 2)
        for group, style in [('heirs', True), ('non_heirs', False)]:
            mask = self.data['is_heir'] if style else ~self.data['is_heir']
            means = [self.data[mask][f'ndvi_{year}'].mean() for year in ANALYSIS_YEARS]
            std = [self.data[mask][f'ndvi_{year}'].std() for year in ANALYSIS_YEARS]
            
            plt.plot(
                ANALYSIS_YEARS,
                means,
                LINE_STYLES[group],
                color=COLOR_SCHEME[group],
                label=group.replace('_', ' ').title(),
                linewidth=2
            )
            plt.fill_between(
                ANALYSIS_YEARS,
                np.array(means) - np.array(std),
                np.array(means) + np.array(std),
                color=COLOR_SCHEME[group],
                alpha=0.2
            )
            
        plt.title('Average NDVI Over Time')
        plt.xlabel('Year')
        plt.ylabel('Mean NDVI')
        plt.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
            logger.info(f"Saved trend comparison plot to {save_path}")
        plt.close()
        
    def plot_ndvi_distributions(self, save_path: Optional[Path] = None) -> None:
        """Create visualization of NDVI distributions for each year.
        
        Args:
            save_path: Optional path to save the plot
        """
        if self.data is None:
            raise ValueError("No data loaded")
            
        fig, axes = plt.subplots(
            1, len(ANALYSIS_YEARS),
            figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1]/2),
            sharey=True
        )
        
        for idx, year in enumerate(ANALYSIS_YEARS):
            for group, style in [('heirs', True), ('non_heirs', False)]:
                mask = self.data['is_heir'] if style else ~self.data['is_heir']
                sns.kdeplot(
                    data=self.data[mask][f'ndvi_{year}'],
                    ax=axes[idx],
                    color=COLOR_SCHEME[group],
                    label=group.replace('_', ' ').title(),
                    linestyle=LINE_STYLES[group]
                )
                
            axes[idx].set_title(f'NDVI Distribution {year}')
            axes[idx].set_xlabel('NDVI Value')
            if idx == 0:
                axes[idx].set_ylabel('Density')
                
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
            logger.info(f"Saved distribution plot to {save_path}")
        plt.close()
        
    def plot_trend_components(
        self,
        save_path: Optional[Path] = None,
        n_samples: int = 100
    ) -> None:
        """Create visualization of trend components and variability.
        
        Args:
            save_path: Optional path to save the plot
            n_samples: Number of individual trends to plot
        """
        if self.data is None:
            raise ValueError("No data loaded")
            
        plt.figure(figsize=FIGURE_SIZE)
        
        for group, style in [('heirs', True), ('non_heirs', False)]:
            mask = self.data['is_heir'] if style else ~self.data['is_heir']
            subset = self.data[mask].sample(
                n=min(n_samples, sum(mask)),
                random_state=42
            )
            
            # Plot individual trends
            for _, row in subset.iterrows():
                values = [row[f'ndvi_{year}'] for year in ANALYSIS_YEARS]
                plt.plot(
                    ANALYSIS_YEARS,
                    values,
                    color=COLOR_SCHEME[group],
                    alpha=0.1,
                    linewidth=0.5
                )
                
            # Plot mean trend
            means = [self.data[mask][f'ndvi_{year}'].mean() for year in ANALYSIS_YEARS]
            plt.plot(
                ANALYSIS_YEARS,
                means,
                LINE_STYLES[group],
                color=COLOR_SCHEME[group],
                label=group.replace('_', ' ').title(),
                linewidth=3
            )
            
        plt.title('NDVI Trends: Individual Properties and Group Means')
        plt.xlabel('Year')
        plt.ylabel('NDVI Value')
        plt.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
            logger.info(f"Saved trend components plot to {save_path}")
        plt.close()
        
    def create_all_plots(self, output_dir: Path) -> Dict[str, Path]:
        """Create all visualization plots and save them.
        
        Args:
            output_dir: Directory to save plots
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        if self.data is None:
            raise ValueError("No data loaded")
            
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_paths = {}
        
        # Trend comparison plot
        trend_path = output_dir / "ndvi_trend_comparison.png"
        self.plot_trend_comparison(trend_path)
        plot_paths['trend_comparison'] = trend_path
        
        # Distribution plot
        dist_path = output_dir / "ndvi_distributions.png"
        self.plot_ndvi_distributions(dist_path)
        plot_paths['distributions'] = dist_path
        
        # Trend components plot
        comp_path = output_dir / "ndvi_trend_components.png"
        self.plot_trend_components(comp_path)
        plot_paths['trend_components'] = comp_path
        
        logger.info(f"Created all plots in {output_dir}")
        return plot_paths 