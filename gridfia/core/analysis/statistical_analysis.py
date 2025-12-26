#!/usr/bin/env python3
"""
Statistical Analysis Utilities

This module provides comprehensive statistical analysis for comparing
species diversity between groups, including modern robust methods,
effect size calculations, and spatial autocorrelation testing.

.. warning::
   This module is currently not integrated into the main analysis pipeline.
   The statistical methods are fully implemented but need to be connected
   to the data processing workflows.

.. todo::
   Integration tasks for statistical analysis:
   
   - [ ] Create generic configuration class to replace removed ComparisonConfig
   - [ ] Add CLI command for statistical analysis workflows
   - [ ] Integrate with species diversity calculations
   - [ ] Add example notebooks demonstrating usage
   - [ ] Create unit tests for all statistical methods
   - [ ] Add support for multiple comparison corrections
   
   Target Version: v0.3.0
   Priority: Medium
   Dependencies: Core calculation pipeline must be complete
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.utils import resample

logger = logging.getLogger(__name__)


@dataclass
class StatisticalConfig:
    """Generic configuration for statistical analysis.
    
    TODO: This is a temporary replacement for the removed ComparisonConfig.
    Should be moved to config.py and integrated with Pydantic models.
    """
    diversity_metrics: List[str] = field(default_factory=lambda: ['richness', 'shannon', 'simpson', 'evenness'])
    bootstrap_iterations: int = 10000
    confidence_level: float = 0.95
    min_sample_size: int = 30
    statistical_tests: List[str] = field(default_factory=lambda: ['mannwhitney', 'permutation', 'bootstrap'])


class DiversityAnalyzer:
    """
    Class for calculating species diversity metrics from spatial data.
    
    Supports multiple diversity indices:
    - Species richness (count of species)
    - Shannon diversity index
    - Simpson's diversity index
    - Pielou's evenness index
    - Chao1 estimator
    - ACE estimator
    """
    
    def __init__(self, config: Optional[StatisticalConfig] = None):
        """
        Initialize the diversity analyzer.
        
        Parameters:
        -----------
        config : StatisticalConfig, optional
            Configuration object with analysis parameters.
            If None, uses default configuration.
        """
        self.config = config or StatisticalConfig()
        self.supported_metrics = {
            'richness', 'shannon', 'simpson', 'evenness', 'chao1', 'ace'
        }
        
        logger.info(f"Initialized DiversityAnalyzer with metrics: {config.diversity_metrics}")
    
    def calculate_richness(self, species_counts: np.ndarray) -> float:
        """Calculate species richness (number of species present)."""
        return np.sum(species_counts > 0)
    
    def calculate_shannon(self, species_counts: np.ndarray) -> float:
        """Calculate Shannon diversity index."""
        # Remove zeros to avoid log(0)
        counts = species_counts[species_counts > 0]
        if len(counts) == 0:
            return 0.0
        
        # Calculate proportions
        proportions = counts / np.sum(counts)
        
        # Shannon index: H = -sum(p_i * log(p_i))
        shannon = -np.sum(proportions * np.log(proportions))
        
        return shannon
    
    def calculate_simpson(self, species_counts: np.ndarray) -> float:
        """Calculate Simpson's diversity index (1 - Simpson's dominance)."""
        total = np.sum(species_counts)
        if total == 0:
            return 0.0
        
        # Simpson's dominance: D = sum(p_i^2)
        proportions = species_counts / total
        dominance = np.sum(proportions ** 2)
        
        # Simpson's diversity: 1 - D
        simpson = 1.0 - dominance
        
        return simpson
    
    def calculate_evenness(self, species_counts: np.ndarray) -> float:
        """Calculate Pielou's evenness index."""
        shannon = self.calculate_shannon(species_counts)
        richness = self.calculate_richness(species_counts)
        
        if richness <= 1:
            return 0.0
        
        # Pielou's evenness: J = H / log(S)
        max_shannon = np.log(richness)
        evenness = shannon / max_shannon if max_shannon > 0 else 0.0
        
        return evenness
    
    def calculate_chao1(self, species_counts: np.ndarray) -> float:
        """Calculate Chao1 estimator for species richness."""
        # Count singletons and doubletons
        singletons = np.sum(species_counts == 1)
        doubletons = np.sum(species_counts == 2)
        observed_richness = self.calculate_richness(species_counts)
        
        if doubletons > 0:
            # Standard Chao1 formula
            chao1 = observed_richness + (singletons ** 2) / (2 * doubletons)
        elif singletons > 0:
            # Modified formula when no doubletons
            chao1 = observed_richness + singletons * (singletons - 1) / 2
        else:
            # No singletons or doubletons
            chao1 = observed_richness
        
        return chao1
    
    def calculate_ace(self, species_counts: np.ndarray, rare_threshold: int = 10) -> float:
        """Calculate ACE (Abundance-based Coverage Estimator)."""
        # Separate rare and abundant species
        rare_mask = (species_counts > 0) & (species_counts <= rare_threshold)
        abundant_mask = species_counts > rare_threshold
        
        n_rare = np.sum(rare_mask)
        n_abundant = np.sum(abundant_mask)
        
        if n_rare == 0:
            return n_abundant
        
        # Calculate coverage estimate
        f1 = np.sum(species_counts == 1)
        n_rare_total = np.sum(species_counts[rare_mask])
        
        if n_rare_total > 0:
            c_ace = 1 - (f1 / n_rare_total)
        else:
            c_ace = 1
        
        if c_ace > 0:
            # Calculate coefficient of variation
            i_values = np.arange(1, rare_threshold + 1)
            f_values = np.array([np.sum(species_counts == i) for i in i_values])
            
            numerator = np.sum(i_values * (i_values - 1) * f_values)
            denominator = n_rare_total * (n_rare_total - 1)
            
            if denominator > 0:
                gamma_ace = max(0, (n_rare / c_ace) * (numerator / denominator) - 1)
            else:
                gamma_ace = 0
            
            # ACE estimate
            ace = n_abundant + (n_rare / c_ace) + (f1 / c_ace) * gamma_ace
        else:
            ace = n_abundant + n_rare
        
        return ace
    
    def calculate_all_metrics(
        self, 
        species_counts: np.ndarray, 
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calculate all requested diversity metrics.
        
        Parameters:
        -----------
        species_counts : np.ndarray
            Array of species counts/abundances
        metrics : list of str, optional
            List of metrics to calculate. If None, uses config metrics.
            
        Returns:
        --------
        dict
            Dictionary with metric names as keys and values as results
        """
        if metrics is None:
            metrics = self.config.diversity_metrics
        
        results = {}
        
        for metric in metrics:
            if metric == 'richness':
                results[metric] = self.calculate_richness(species_counts)
            elif metric == 'shannon':
                results[metric] = self.calculate_shannon(species_counts)
            elif metric == 'simpson':
                results[metric] = self.calculate_simpson(species_counts)
            elif metric == 'evenness':
                results[metric] = self.calculate_evenness(species_counts)
            elif metric == 'chao1':
                results[metric] = self.calculate_chao1(species_counts)
            elif metric == 'ace':
                results[metric] = self.calculate_ace(species_counts)
            else:
                logger.warning(f"Unknown diversity metric: {metric}")
                results[metric] = np.nan
        
        return results


class StatisticalTester:
    """
    Class for performing statistical comparisons between groups.
    
    Provides multiple statistical tests with appropriate corrections
    for multiple comparisons and spatial data characteristics.
    """
    
    def __init__(self, config: Optional[StatisticalConfig] = None):
        """
        Initialize the statistical tester.
        
        Parameters:
        -----------
        config : StatisticalConfig, optional
            Configuration object with analysis parameters.
            If None, uses default configuration.
        """
        self.config = config or StatisticalConfig()
        self.alpha = 1 - self.config.confidence_level
        
        logger.info(f"Initialized StatisticalTester with alpha: {self.alpha}")
    
    def compare_groups(
        self,
        data: pd.DataFrame,
        group_column: str,
        metric_columns: List[str]
    ) -> Dict[str, Dict]:
        """
        Compare groups across multiple metrics using various statistical tests.
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing diversity metrics and group labels
        group_column : str
            Name of column containing group labels
        metric_columns : list of str
            List of metric columns to compare
            
        Returns:
        --------
        dict
            Nested dictionary with test results for each metric
        """
        results = {}
        
        # Get unique groups
        groups = data[group_column].unique()
        if len(groups) != 2:
            raise ValueError(f"Expected 2 groups, found {len(groups)}: {groups}")
        
        group1_label, group2_label = groups
        
        for metric in metric_columns:
            logger.info(f"Analyzing metric: {metric}")
            
            # Extract data for each group
            group1_data = data[data[group_column] == group1_label][metric].dropna()
            group2_data = data[data[group_column] == group2_label][metric].dropna()
            
            if len(group1_data) == 0 or len(group2_data) == 0:
                logger.warning(f"Insufficient data for metric {metric}")
                results[metric] = {'error': 'Insufficient data'}
                continue
            
            # Perform multiple statistical tests
            metric_results = {}
            
            # Descriptive statistics
            metric_results['descriptive'] = {
                f'{group1_label}_mean': group1_data.mean(),
                f'{group1_label}_std': group1_data.std(),
                f'{group1_label}_n': len(group1_data),
                f'{group2_label}_mean': group2_data.mean(),
                f'{group2_label}_std': group2_data.std(),
                f'{group2_label}_n': len(group2_data),
                'difference': group1_data.mean() - group2_data.mean()
            }
            
            # Statistical tests
            if 'mannwhitney' in self.config.statistical_tests:
                metric_results['mannwhitney'] = self._mann_whitney_test(group1_data, group2_data)
            
            if 'permutation' in self.config.statistical_tests:
                metric_results['permutation'] = self._permutation_test(group1_data, group2_data)
            
            if 'bootstrap' in self.config.statistical_tests:
                metric_results['bootstrap'] = self._bootstrap_test(group1_data, group2_data)
            
            # Effect size calculations
            metric_results['effect_size'] = self._calculate_effect_sizes(group1_data, group2_data)
            
            results[metric] = metric_results
        
        # Apply multiple comparison correction
        results = self._apply_multiple_comparison_correction(results)
        
        return results
    
    def _mann_whitney_test(
        self, 
        group1: pd.Series, 
        group2: pd.Series
    ) -> Dict[str, float]:
        """Perform Mann-Whitney U test (non-parametric)."""
        try:
            statistic, p_value = stats.mannwhitneyu(
                group1, group2, 
                alternative='two-sided'
            )
            
            return {
                'statistic': statistic,
                'p_value': p_value,
                'test_type': 'mann_whitney_u'
            }
        except Exception as e:
            logger.error(f"Mann-Whitney test failed: {e}")
            return {'error': str(e)}
    
    def _permutation_test(
        self, 
        group1: pd.Series, 
        group2: pd.Series,
        n_permutations: int = 10000
    ) -> Dict[str, float]:
        """Perform permutation test for difference in means with optional parallel processing."""
        try:
            # Use parallel processing for large permutation tests
            if n_permutations > 5000:
                try:
                    from .parallel_processing import parallel_permutation_test
                    
                    logger.debug(f"Using parallel permutation test with {n_permutations} iterations")
                    
                    results = parallel_permutation_test(
                        group1.values,
                        group2.values,
                        n_permutations=n_permutations
                    )
                    
                    if 'error' not in results:
                        results['test_type'] = 'permutation_parallel'
                        return results
                    else:
                        logger.warning("Parallel permutation test failed, using sequential")
                        
                except ImportError:
                    logger.debug("Parallel processing not available for permutation test")
            
            # Sequential permutation test implementation
            # Observed difference
            observed_diff = group1.mean() - group2.mean()
            
            # Combine all data
            combined = np.concatenate([group1.values, group2.values])
            n1, n2 = len(group1), len(group2)
            
            # Permutation distribution
            perm_diffs = []
            for _ in range(n_permutations):
                # Randomly permute the combined data
                np.random.shuffle(combined)
                
                # Split into two groups of original sizes
                perm_group1 = combined[:n1]
                perm_group2 = combined[n1:n1+n2]
                
                # Calculate difference
                perm_diff = perm_group1.mean() - perm_group2.mean()
                perm_diffs.append(perm_diff)
            
            perm_diffs = np.array(perm_diffs)
            
            # Calculate p-value (two-tailed)
            p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
            
            return {
                'observed_difference': observed_diff,
                'p_value': p_value,
                'n_permutations': n_permutations,
                'test_type': 'permutation'
            }
        except Exception as e:
            logger.error(f"Permutation test failed: {e}")
            return {'error': str(e)}
    
    def _bootstrap_test(
        self, 
        group1: pd.Series, 
        group2: pd.Series
    ) -> Dict[str, float]:
        """Perform bootstrap test for confidence intervals with optional parallel processing."""
        try:
            n_bootstrap = self.config.bootstrap_iterations
            
            # Use parallel processing for large bootstrap iterations
            if n_bootstrap > 5000:
                try:
                    from .parallel_processing import parallel_bootstrap_analysis
                    
                    logger.debug(f"Using parallel bootstrap with {n_bootstrap} iterations")
                    
                    results = parallel_bootstrap_analysis(
                        group1.values,
                        group2.values,
                        n_bootstrap=n_bootstrap
                    )
                    
                    if 'error' not in results:
                        return {
                            'difference_ci_lower': results['ci_lower'],
                            'difference_ci_upper': results['ci_upper'],
                            'significant': results['significant'],
                            'test_type': 'bootstrap_parallel',
                            'confidence_level': self.config.confidence_level,
                            'n_bootstrap_actual': results['n_bootstrap']
                        }
                    else:
                        logger.warning("Parallel bootstrap failed, using sequential")
                        
                except ImportError:
                    logger.debug("Parallel processing not available for bootstrap")
            
            # Sequential bootstrap implementation
            # Bootstrap distributions
            group1_boots = []
            group2_boots = []
            diff_boots = []
            
            for _ in range(n_bootstrap):
                # Bootstrap samples
                boot1 = resample(group1.values, n_samples=len(group1))
                boot2 = resample(group2.values, n_samples=len(group2))
                
                mean1 = np.mean(boot1)
                mean2 = np.mean(boot2)
                
                group1_boots.append(mean1)
                group2_boots.append(mean2)
                diff_boots.append(mean1 - mean2)
            
            # Calculate confidence intervals
            alpha = self.alpha
            ci_lower = alpha / 2
            ci_upper = 1 - alpha / 2
            
            group1_ci = np.percentile(group1_boots, [ci_lower * 100, ci_upper * 100])
            group2_ci = np.percentile(group2_boots, [ci_lower * 100, ci_upper * 100])
            diff_ci = np.percentile(diff_boots, [ci_lower * 100, ci_upper * 100])
            
            return {
                'group1_ci_lower': group1_ci[0],
                'group1_ci_upper': group1_ci[1],
                'group2_ci_lower': group2_ci[0],
                'group2_ci_upper': group2_ci[1],
                'difference_ci_lower': diff_ci[0],
                'difference_ci_upper': diff_ci[1],
                'significant': not (diff_ci[0] <= 0 <= diff_ci[1]),
                'test_type': 'bootstrap',
                'confidence_level': self.config.confidence_level
            }
        except Exception as e:
            logger.error(f"Bootstrap test failed: {e}")
            return {'error': str(e)}
    
    def _calculate_effect_sizes(
        self, 
        group1: pd.Series, 
        group2: pd.Series
    ) -> Dict[str, float]:
        """Calculate various effect size measures."""
        try:
            # Cohen's d
            pooled_std = np.sqrt(((len(group1) - 1) * group1.var() + 
                                 (len(group2) - 1) * group2.var()) / 
                                (len(group1) + len(group2) - 2))
            
            cohens_d = (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0
            
            # Glass's delta (using group2 as control)
            glass_delta = (group1.mean() - group2.mean()) / group2.std() if group2.std() > 0 else 0
            
            # Hedges' g (bias-corrected Cohen's d)
            j_factor = 1 - (3 / (4 * (len(group1) + len(group2)) - 9))
            hedges_g = cohens_d * j_factor
            
            # Cliff's delta (non-parametric effect size)
            cliffs_delta = self._calculate_cliffs_delta(group1, group2)
            
            return {
                'cohens_d': cohens_d,
                'glass_delta': glass_delta,
                'hedges_g': hedges_g,
                'cliffs_delta': cliffs_delta
            }
        except Exception as e:
            logger.error(f"Effect size calculation failed: {e}")
            return {'error': str(e)}
    
    def _calculate_cliffs_delta(self, group1: pd.Series, group2: pd.Series) -> float:
        """Calculate Cliff's delta (non-parametric effect size)."""
        n1, n2 = len(group1), len(group2)
        
        # Count pairs where group1 > group2, group1 < group2
        greater = 0
        less = 0
        
        for x1 in group1:
            for x2 in group2:
                if x1 > x2:
                    greater += 1
                elif x1 < x2:
                    less += 1
        
        # Cliff's delta
        delta = (greater - less) / (n1 * n2)
        
        return delta
    
    def _apply_multiple_comparison_correction(
        self, 
        results: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """Apply multiple comparison correction (Benjamini-Hochberg)."""
        try:
            # Collect all p-values
            p_values = []
            p_value_keys = []
            
            for metric, metric_results in results.items():
                for test_name, test_results in metric_results.items():
                    if isinstance(test_results, dict) and 'p_value' in test_results:
                        p_values.append(test_results['p_value'])
                        p_value_keys.append((metric, test_name))
            
            if len(p_values) == 0:
                return results
            
            # Apply Benjamini-Hochberg correction
            corrected_p = self._benjamini_hochberg_correction(p_values)
            
            # Update results with corrected p-values
            for i, (metric, test_name) in enumerate(p_value_keys):
                results[metric][test_name]['p_value_corrected'] = corrected_p[i]
                results[metric][test_name]['significant_corrected'] = corrected_p[i] < self.alpha
            
            return results
            
        except Exception as e:
            logger.error(f"Multiple comparison correction failed: {e}")
            return results
    
    def _benjamini_hochberg_correction(self, p_values: List[float]) -> List[float]:
        """Apply Benjamini-Hochberg (FDR) correction."""
        p_values = np.array(p_values)
        n = len(p_values)
        
        # Sort p-values and keep track of original indices
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        
        # Apply correction
        corrected_p = np.zeros(n)
        for i in range(n):
            corrected_p[i] = min(1.0, sorted_p[i] * n / (i + 1))
        
        # Ensure monotonicity
        for i in range(n - 2, -1, -1):
            corrected_p[i] = min(corrected_p[i], corrected_p[i + 1])
        
        # Restore original order
        result = np.zeros(n)
        result[sorted_indices] = corrected_p
        
        return result.tolist()


def compute_spatial_autocorrelation(
    data: pd.DataFrame,
    geometry_column: str = 'geometry',
    value_column: str = 'value'
) -> Dict[str, float]:
    """
    Compute spatial autocorrelation using Moran's I.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with geometry and values
    geometry_column : str
        Name of geometry column
    value_column : str
        Name of value column to test
        
    Returns:
    --------
    dict
        Moran's I test results
    """
    try:
        import libpysal
        from esda.moran import Moran
        
        # Create spatial weights matrix
        weights = libpysal.weights.Queen.from_dataframe(data, geom_col=geometry_column)
        
        # Calculate Moran's I
        moran = Moran(data[value_column], weights)
        
        return {
            'morans_i': moran.I,
            'expected_i': moran.EI,
            'variance_i': moran.VI_norm,
            'z_score': moran.z_norm,
            'p_value': moran.p_norm,
            'significant': moran.p_norm < 0.05
        }
        
    except ImportError:
        logger.warning("libpysal not available, skipping spatial autocorrelation test")
        return {'error': 'libpysal not available'}
    except Exception as e:
        logger.error(f"Spatial autocorrelation test failed: {e}")
        return {'error': str(e)} 