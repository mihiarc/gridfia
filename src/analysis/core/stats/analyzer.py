#!/usr/bin/env python3

from dataclasses import dataclass
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

@dataclass
class StatsConfig:
    """Configuration for statistical analysis."""
    confidence_level: float = 0.95
    min_sample_size: int = 30
    methods: List[str] = ('t_test', 'mann_whitney')

class StatsAnalyzer:
    """Analyzes statistical differences between heirs and non-heirs properties."""
    
    def __init__(self, config: StatsConfig):
        """Initialize the statistical analyzer.
        
        Args:
            config: StatsConfig object with analysis parameters
        """
        self.config = config
        
    def analyze_ndvi_differences(self,
                               heirs_ndvi: pd.DataFrame,
                               matches_ndvi: pd.DataFrame) -> Dict:
        """Analyze statistical differences in NDVI between heirs and matches.
        
        Args:
            heirs_ndvi: DataFrame with NDVI data for heirs properties
            matches_ndvi: DataFrame with NDVI data for matched properties
            
        Returns:
            Dictionary with statistical results
        """
        results = {}
        
        # Basic statistics
        results['descriptive'] = self._calculate_descriptive_stats(
            heirs_ndvi, matches_ndvi
        )
        
        # Statistical tests
        if len(heirs_ndvi) >= self.config.min_sample_size and \
           len(matches_ndvi) >= self.config.min_sample_size:
            
            results['tests'] = self._run_statistical_tests(
                heirs_ndvi['mean_ndvi'].values,
                matches_ndvi['mean_ndvi'].values
            )
        else:
            logger.warning("Sample size too small for statistical tests")
            results['tests'] = None
            
        # Effect sizes
        results['effect_sizes'] = self._calculate_effect_sizes(
            heirs_ndvi['mean_ndvi'].values,
            matches_ndvi['mean_ndvi'].values
        )
        
        return results
    
    def _calculate_descriptive_stats(self,
                                   heirs_ndvi: pd.DataFrame,
                                   matches_ndvi: pd.DataFrame) -> Dict:
        """Calculate descriptive statistics.
        
        Args:
            heirs_ndvi: DataFrame with NDVI data for heirs properties
            matches_ndvi: DataFrame with NDVI data for matched properties
            
        Returns:
            Dictionary with descriptive statistics
        """
        stats = {
            'heirs': {
                'count': len(heirs_ndvi),
                'mean': float(heirs_ndvi['mean_ndvi'].mean()),
                'median': float(heirs_ndvi['mean_ndvi'].median()),
                'std': float(heirs_ndvi['mean_ndvi'].std()),
                'min': float(heirs_ndvi['mean_ndvi'].min()),
                'max': float(heirs_ndvi['mean_ndvi'].max())
            },
            'matches': {
                'count': len(matches_ndvi),
                'mean': float(matches_ndvi['mean_ndvi'].mean()),
                'median': float(matches_ndvi['mean_ndvi'].median()),
                'std': float(matches_ndvi['mean_ndvi'].std()),
                'min': float(matches_ndvi['mean_ndvi'].min()),
                'max': float(matches_ndvi['mean_ndvi'].max())
            }
        }
        
        return stats
    
    def _run_statistical_tests(self,
                             heirs_values: np.ndarray,
                             matches_values: np.ndarray) -> Dict:
        """Run statistical tests comparing heirs and matches.
        
        Args:
            heirs_values: NDVI values for heirs properties
            matches_values: NDVI values for matched properties
            
        Returns:
            Dictionary with test results
        """
        results = {}
        
        if 't_test' in self.config.methods:
            t_stat, p_value = stats.ttest_ind(heirs_values, matches_values)
            results['t_test'] = {
                'statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': float(p_value) < (1 - self.config.confidence_level)
            }
            
        if 'mann_whitney' in self.config.methods:
            u_stat, p_value = stats.mannwhitneyu(heirs_values, matches_values)
            results['mann_whitney'] = {
                'statistic': float(u_stat),
                'p_value': float(p_value),
                'significant': float(p_value) < (1 - self.config.confidence_level)
            }
            
        return results
    
    def _calculate_effect_sizes(self,
                              heirs_values: np.ndarray,
                              matches_values: np.ndarray) -> Dict:
        """Calculate effect sizes for the difference between groups.
        
        Args:
            heirs_values: NDVI values for heirs properties
            matches_values: NDVI values for matched properties
            
        Returns:
            Dictionary with effect sizes
        """
        # Cohen's d
        pooled_std = np.sqrt((np.var(heirs_values) + np.var(matches_values)) / 2)
        cohens_d = (np.mean(heirs_values) - np.mean(matches_values)) / pooled_std
        
        # Percent difference
        percent_diff = ((np.mean(heirs_values) - np.mean(matches_values)) / 
                       np.mean(matches_values)) * 100
        
        return {
            'cohens_d': float(cohens_d),
            'percent_difference': float(percent_diff)
        }
    
    def generate_plots(self,
                      heirs_ndvi: pd.DataFrame,
                      matches_ndvi: pd.DataFrame,
                      output_dir: Path = Path("data/processed/plots")) -> None:
        """Generate visualization plots for NDVI comparison.
        
        Args:
            heirs_ndvi: DataFrame with NDVI data for heirs properties
            matches_ndvi: DataFrame with NDVI data for matched properties
            output_dir: Directory to save plots
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Distribution plot
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=heirs_ndvi['mean_ndvi'], label='Heirs Properties')
        sns.kdeplot(data=matches_ndvi['mean_ndvi'], label='Matched Properties')
        plt.title('Distribution of NDVI Values')
        plt.xlabel('NDVI')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(output_dir / 'ndvi_distribution.png')
        plt.close()
        
        # Box plot
        plt.figure(figsize=(8, 6))
        data = pd.concat([
            pd.DataFrame({'NDVI': heirs_ndvi['mean_ndvi'], 'Group': 'Heirs'}),
            pd.DataFrame({'NDVI': matches_ndvi['mean_ndvi'], 'Group': 'Matches'})
        ])
        sns.boxplot(x='Group', y='NDVI', data=data)
        plt.title('NDVI Comparison')
        plt.savefig(output_dir / 'ndvi_boxplot.png')
        plt.close() 