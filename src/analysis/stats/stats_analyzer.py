"""
Vance County NDVI Statistical Analysis Module

This module handles the statistical analysis of NDVI data for Vance County properties,
comparing heirs and non-heirs properties. It includes trend analysis, statistical
testing, and basic summary statistics.
"""
import pandas as pd
import numpy as np
from scipy import stats
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

from ..config.analysis_config import (
    ANALYSIS_YEARS,
    SIGNIFICANCE_LEVEL,
    MIN_SAMPLES,
    LOG_FORMAT,
    STANDARD_CRS
)

# Configure logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class NDVIStatsAnalyzer:
    """Handles statistical analysis of NDVI data for Vance County properties."""
    
    def __init__(self, ndvi_data: Optional[pd.DataFrame] = None):
        """Initialize the NDVI stats analyzer.
        
        Args:
            ndvi_data: Optional DataFrame containing NDVI data
        """
        self.data = ndvi_data
        
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
            
    def validate_data(self) -> bool:
        """Validate loaded data for analysis requirements.
        
        Returns:
            True if data is valid, False otherwise
        """
        if self.data is None:
            logger.error("No data loaded")
            return False
            
        # Check required columns
        required_columns = ['is_heir', 'ndvi_trend_slope'] + \
                         [f'ndvi_{year}' for year in ANALYSIS_YEARS]
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
            
        # Check sample sizes
        heirs_count = self.data['is_heir'].sum()
        non_heirs_count = (~self.data['is_heir']).sum()
        
        if heirs_count < MIN_SAMPLES or non_heirs_count < MIN_SAMPLES:
            logger.warning(
                f"Small sample size: {heirs_count} heirs, {non_heirs_count} non-heirs"
            )
            
        return True
            
    def calculate_basic_statistics(self) -> Dict[str, pd.DataFrame]:
        """Calculate basic NDVI statistics for both groups.
        
        Returns:
            Dictionary containing statistics for each group
        """
        if not self.validate_data():
            raise ValueError("Invalid data for analysis")
            
        stats_dict = {}
        
        # Calculate yearly statistics
        for group in ['heirs', 'non_heirs']:
            mask = self.data['is_heir'] if group == 'heirs' else ~self.data['is_heir']
            yearly_stats = []
            
            for year in ANALYSIS_YEARS:
                stats = self.data[mask][f'ndvi_{year}'].describe()
                yearly_stats.append(pd.DataFrame(stats, columns=[str(year)]))
                
            stats_dict[group] = pd.concat(yearly_stats, axis=1)
            
        return stats_dict
        
    def calculate_trend_statistics(self) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Calculate trend statistics and perform statistical tests.
        
        Returns:
            Tuple of (trend statistics DataFrame, test results dictionary)
        """
        if not self.validate_data():
            raise ValueError("Invalid data for analysis")
            
        # Calculate basic trend statistics
        stats_dict = {
            'heirs': self.data[self.data['is_heir']]['ndvi_trend_slope'].describe(),
            'non_heirs': self.data[~self.data['is_heir']]['ndvi_trend_slope'].describe()
        }
        stats_df = pd.DataFrame(stats_dict)
        
        # Perform statistical tests
        heirs_slope = self.data[self.data['is_heir']]['ndvi_trend_slope']
        non_heirs_slope = self.data[~self.data['is_heir']]['ndvi_trend_slope']
        
        # T-test for trend differences
        t_stat, p_value = stats.ttest_ind(heirs_slope, non_heirs_slope)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((heirs_slope.var() + non_heirs_slope.var()) / 2)
        cohens_d = (heirs_slope.mean() - non_heirs_slope.mean()) / pooled_std
        
        # Add test statistics to DataFrame
        stats_df.loc['t_statistic'] = t_stat
        stats_df.loc['p_value'] = p_value
        stats_df.loc['cohens_d'] = cohens_d
        
        # Prepare test results
        test_results = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'significant': p_value < SIGNIFICANCE_LEVEL
        }
        
        return stats_df, test_results
        
    def calculate_yearly_differences(self) -> pd.DataFrame:
        """Calculate year-by-year differences between groups.
        
        Returns:
            DataFrame containing yearly differences and statistics
        """
        if not self.validate_data():
            raise ValueError("Invalid data for analysis")
            
        differences = []
        for year in ANALYSIS_YEARS:
            heirs = self.data[self.data['is_heir']][f'ndvi_{year}']
            non_heirs = self.data[~self.data['is_heir']][f'ndvi_{year}']
            
            # Calculate difference statistics
            mean_diff = heirs.mean() - non_heirs.mean()
            t_stat, p_value = stats.ttest_ind(heirs, non_heirs)
            
            # Effect size
            pooled_std = np.sqrt((heirs.var() + non_heirs.var()) / 2)
            cohens_d = (heirs.mean() - non_heirs.mean()) / pooled_std
            
            differences.append({
                'year': year,
                'mean_difference': mean_diff,
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significant': p_value < SIGNIFICANCE_LEVEL
            })
            
        return pd.DataFrame(differences)
        
    def generate_summary_report(self) -> str:
        """Generate a markdown summary report of the analysis.
        
        Returns:
            Markdown formatted summary report
        """
        if not self.validate_data():
            raise ValueError("Invalid data for analysis")
            
        # Get statistics
        basic_stats = self.calculate_basic_statistics()
        trend_stats, trend_tests = self.calculate_trend_statistics()
        yearly_diff = self.calculate_yearly_differences()
        
        # Build report
        report = ["# NDVI Analysis Summary Report\n"]
        
        # Sample sizes
        heirs_count = self.data['is_heir'].sum()
        non_heirs_count = (~self.data['is_heir']).sum()
        report.append("## Sample Sizes")
        report.append(f"- Heirs Properties: {heirs_count}")
        report.append(f"- Non-Heirs Properties: {non_heirs_count}\n")
        
        # Trend analysis
        report.append("## Trend Analysis")
        report.append("### Overall Trends")
        report.append("| Statistic | Heirs | Non-Heirs |")
        report.append("|-----------|--------|------------|")
        for stat in ['mean', 'std', 'min', 'max']:
            report.append(
                f"| {stat.capitalize()} | {trend_stats[stat]['heirs']:.4f} | "
                f"{trend_stats[stat]['non_heirs']:.4f} |"
            )
            
        # Statistical tests
        report.append("\n### Statistical Tests")
        report.append(f"- T-statistic: {trend_tests['t_statistic']:.4f}")
        report.append(f"- P-value: {trend_tests['p_value']:.4f}")
        report.append(f"- Cohen's d: {trend_tests['cohens_d']:.4f}")
        report.append(
            f"- Significance: {'Significant' if trend_tests['significant'] else 'Not significant'} "
            f"at α={SIGNIFICANCE_LEVEL}"
        )
        
        # Yearly differences
        report.append("\n## Yearly Differences")
        report.append("| Year | Mean Difference | T-statistic | P-value | Cohen's d | Significant |")
        report.append("|------|-----------------|-------------|----------|-----------|-------------|")
        for _, row in yearly_diff.iterrows():
            report.append(
                f"| {row['year']} | {row['mean_difference']:.4f} | "
                f"{row['t_statistic']:.4f} | {row['p_value']:.4f} | "
                f"{row['cohens_d']:.4f} | {'Yes' if row['significant'] else 'No'} |"
            )
            
        return "\n".join(report) 