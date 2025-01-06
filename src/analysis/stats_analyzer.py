import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StatsAnalyzer:
    def __init__(self, output_dir: str = "output/analysis"):
        """Initialize the StatsAnalyzer with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_ndvi_data(self, ndvi_file: str) -> pd.DataFrame:
        """Load NDVI data from processed parquet file."""
        try:
            df = pd.read_parquet(ndvi_file)
            logger.info(f"Loaded NDVI data with {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Error loading NDVI data: {e}")
            raise
            
    def calculate_trend_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend statistics for heirs vs non-heirs properties."""
        stats_dict = {
            'heirs': df[df['is_heir']]['ndvi_trend_slope'].describe(),
            'non_heirs': df[~df['is_heir']]['ndvi_trend_slope'].describe()
        }
        
        # Perform t-test between groups
        heirs_slope = df[df['is_heir']]['ndvi_trend_slope']
        non_heirs_slope = df[~df['is_heir']]['ndvi_trend_slope']
        t_stat, p_value = stats.ttest_ind(heirs_slope, non_heirs_slope)
        
        stats_df = pd.DataFrame(stats_dict)
        stats_df.loc['t_statistic'] = t_stat
        stats_df.loc['p_value'] = p_value
        
        return stats_df
        
    def plot_trend_comparison(self, df: pd.DataFrame, save_path: str):
        """Create visualization comparing NDVI trends between groups."""
        plt.figure(figsize=(12, 6))
        
        # Trend slope distribution
        plt.subplot(1, 2, 1)
        sns.boxplot(data=df, x='is_heir', y='ndvi_trend_slope')
        plt.title('NDVI Trend Slope Distribution')
        plt.xlabel('Heirs Property')
        plt.ylabel('Trend Slope')
        
        # NDVI values over time
        plt.subplot(1, 2, 2)
        years = [2018, 2020, 2022]
        heirs_means = [df[df['is_heir']][f'ndvi_{year}'].mean() for year in years]
        non_heirs_means = [df[~df['is_heir']][f'ndvi_{year}'].mean() for year in years]
        
        plt.plot(years, heirs_means, 'b-', label='Heirs Properties')
        plt.plot(years, non_heirs_means, 'r--', label='Non-Heirs Properties')
        plt.title('Average NDVI Over Time')
        plt.xlabel('Year')
        plt.ylabel('Mean NDVI')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    def run_analysis(self, ndvi_file: str):
        """Run complete analysis workflow."""
        try:
            # Load and process data
            df = self.load_ndvi_data(ndvi_file)
            
            # Calculate statistics
            stats_df = self.calculate_trend_statistics(df)
            stats_output = self.output_dir / 'ndvi_comparison_stats.csv'
            stats_df.to_csv(stats_output)
            logger.info(f"Saved comparison statistics to {stats_output}")
            
            # Create visualizations
            plot_output = self.output_dir / 'ndvi_trend_comparison.png'
            self.plot_trend_comparison(df, str(plot_output))
            logger.info(f"Saved comparison plot to {plot_output}")
            
            return stats_df
            
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            raise 