#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

class StatsAnalyzer:
    """Analyzes NDVI statistics for heirs and non-heirs properties."""
    
    def __init__(self, output_dir: str = "output/analysis"):
        """Initialize the stats analyzer.
        
        Args:
            output_dir: Directory to save analysis results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.years = ['2018', '2020', '2022']
        
    def calculate_statistics(self, matches_df: pd.DataFrame, ndvi_df: pd.DataFrame) -> Dict:
        """Calculate statistics comparing heirs and non-heirs properties.
        
        Args:
            matches_df: DataFrame with property matches
            ndvi_df: DataFrame with NDVI values
            
        Returns:
            Dictionary containing statistical results
            
        Raises:
            KeyError: If required columns are missing
            ValueError: If NDVI values are invalid or out of range
        """
        # Handle empty data first
        if len(matches_df) == 0 or len(ndvi_df) == 0:
            return {
                'sample_size': 0,
                'years_analyzed': self.years,
                'comparisons': {}
            }
        
        # Validate input data
        required_columns = ['property_id', 'property_type']
        if not all(col in ndvi_df.columns for col in required_columns):
            raise KeyError("Missing required columns in NDVI data")
        
        # Validate NDVI values
        for year in self.years:
            col = f'ndvi_{year}'
            if col in ndvi_df.columns:
                if not pd.api.types.is_numeric_dtype(ndvi_df[col]):
                    raise ValueError("Invalid NDVI values")
                valid_values = ndvi_df[col].dropna()
                if len(valid_values) > 0:
                    if (valid_values < -1).any() or (valid_values > 1).any():
                        raise ValueError("NDVI values out of range")
        
        stats_dict = {
            'sample_size': len(matches_df),
            'years_analyzed': self.years,
            'comparisons': {}
        }
        
        # Merge matches with NDVI data
        try:
            analysis_df = matches_df.merge(
                ndvi_df.add_prefix('heir_'),
                left_on='heir_id',
                right_on='heir_property_id'
            ).merge(
                ndvi_df.add_prefix('neighbor_'),
                left_on='neighbor_id',
                right_on='neighbor_property_id'
            )
        except Exception as e:
            raise ValueError(f"Error merging data: {str(e)}")
        
        for year in self.years:
            heir_col = f'heir_ndvi_{year}'
            neighbor_col = f'neighbor_ndvi_{year}'
            
            if heir_col in analysis_df.columns and neighbor_col in analysis_df.columns:
                # Validate NDVI values
                for col in [heir_col, neighbor_col]:
                    if not pd.api.types.is_numeric_dtype(analysis_df[col]):
                        raise ValueError("Invalid NDVI values")
                    # Check NDVI range
                    valid_values = analysis_df[col].dropna()
                    if len(valid_values) > 0:
                        if (valid_values < -1).any() or (valid_values > 1).any():
                            raise ValueError("NDVI values out of range")
                
                # Get valid pairs
                valid_pairs = analysis_df[[heir_col, neighbor_col]].dropna()
                
                if len(valid_pairs) > 0:
                    # Calculate differences
                    differences = valid_pairs[heir_col] - valid_pairs[neighbor_col]
                    
                    # Perform t-test
                    t_stat, p_value = stats.ttest_rel(
                        valid_pairs[heir_col],
                        valid_pairs[neighbor_col]
                    )
                    
                    # Store statistics
                    stats_dict['comparisons'][year] = {
                        'sample_size': len(valid_pairs),
                        'heirs_mean': float(valid_pairs[heir_col].mean()),
                        'heirs_std': float(valid_pairs[heir_col].std()),
                        'neighbors_mean': float(valid_pairs[neighbor_col].mean()),
                        'neighbors_std': float(valid_pairs[neighbor_col].std()),
                        'mean_difference': float(differences.mean()),
                        'difference_std': float(differences.std()),
                        't_statistic': float(t_stat),
                        'p_value': float(p_value)
                    }
        
        return stats_dict
    
    def create_visualizations(self, matches_df: pd.DataFrame, ndvi_df: pd.DataFrame):
        """Create visualizations of the NDVI comparisons."""
        print("\nPreparing visualization data...")
        
        # Debug info about input data
        print(f"\nMatches DataFrame shape: {matches_df.shape}")
        print(f"NDVI DataFrame shape: {ndvi_df.shape}")
        print("\nMatches DataFrame columns:", matches_df.columns.tolist())
        print("NDVI DataFrame columns:", ndvi_df.columns.tolist())
        
        # Debug property IDs
        print("\nSample of heir_ids from matches:", matches_df['heir_id'].head().tolist())
        print("Sample of neighbor_ids from matches:", matches_df['neighbor_id'].head().tolist())
        print("Sample of property_ids from NDVI:", ndvi_df['property_id'].head().tolist())
        
        # Merge data
        print("\nMerging data...")
        
        # First, merge heir properties
        heir_merge = matches_df.merge(
            ndvi_df,
            left_on='heir_id',
            right_on='property_id',
            how='inner'
        )
        print(f"After heir merge: {heir_merge.shape[0]} rows")
        
        # Rename columns to distinguish heir NDVI values
        heir_cols = {
            'property_type': 'heir_property_type',
            'ndvi_2018': 'heir_ndvi_2018',
            'ndvi_2020': 'heir_ndvi_2020',
            'ndvi_2022': 'heir_ndvi_2022'
        }
        heir_merge = heir_merge.rename(columns=heir_cols)
        
        # Then merge neighbor properties
        analysis_df = heir_merge.merge(
            ndvi_df,
            left_on='neighbor_id',
            right_on='property_id',
            how='inner',
            suffixes=('', '_neighbor')
        )
        print(f"After neighbor merge: {analysis_df.shape[0]} rows")
        
        # Rename neighbor columns
        neighbor_cols = {
            'property_type': 'neighbor_property_type',
            'ndvi_2018': 'neighbor_ndvi_2018',
            'ndvi_2020': 'neighbor_ndvi_2020',
            'ndvi_2022': 'neighbor_ndvi_2022'
        }
        analysis_df = analysis_df.rename(columns=neighbor_cols)
        
        print(f"\nMerged analysis DataFrame shape: {analysis_df.shape}")
        print("Analysis DataFrame columns:", analysis_df.columns.tolist())
        
        # Save analysis data to CSV
        print("\nSaving analysis data...")
        analysis_df.to_csv(self.output_dir / 'analysis_data.csv', index=False)
        
        # Calculate time series data
        print("\nCalculating time series data...")
        time_series_data = {
            'year': self.years,
            'heir_mean': [],
            'heir_std': [],
            'neighbor_mean': [],
            'neighbor_std': []
        }
        
        for year in self.years:
            heir_col = f'heir_ndvi_{year}'
            neighbor_col = f'neighbor_ndvi_{year}'
            
            if heir_col in analysis_df.columns and neighbor_col in analysis_df.columns:
                heir_values = analysis_df[heir_col].dropna()
                neighbor_values = analysis_df[neighbor_col].dropna()
                
                print(f"\nYear {year}:")
                print(f"Heir values: {len(heir_values)} (mean: {heir_values.mean():.4f})")
                print(f"Neighbor values: {len(neighbor_values)} (mean: {neighbor_values.mean():.4f})")
                
                time_series_data['heir_mean'].append(heir_values.mean())
                time_series_data['heir_std'].append(heir_values.std())
                time_series_data['neighbor_mean'].append(neighbor_values.mean())
                time_series_data['neighbor_std'].append(neighbor_values.std())
        
        # Save time series data
        print("\nSaving time series data...")
        pd.DataFrame(time_series_data).to_csv(self.output_dir / 'time_series_data.csv', index=False)
        
        # Calculate differences for boxplots
        print("\nCalculating differences data...")
        differences_data = {
            'year': [],
            'ndvi_difference': []
        }
        
        for year in self.years:
            heir_col = f'heir_ndvi_{year}'
            neighbor_col = f'neighbor_ndvi_{year}'
            
            if heir_col in analysis_df.columns and neighbor_col in analysis_df.columns:
                diff = analysis_df[heir_col].fillna(0) - analysis_df[neighbor_col].fillna(0)
                valid_diff = diff[diff != 0]  # Remove pairs where both values were NA
                
                print(f"\nYear {year} differences:")
                print(f"Total pairs: {len(diff)}")
                print(f"Valid differences: {len(valid_diff)}")
                print(f"Mean difference: {valid_diff.mean():.4f}")
                
                differences_data['year'].extend([year] * len(valid_diff))
                differences_data['ndvi_difference'].extend(valid_diff.tolist())
        
        # Save differences data
        print("\nSaving differences data...")
        pd.DataFrame(differences_data).to_csv(self.output_dir / 'differences_data.csv', index=False)

        # Time series plot
        plt.figure(figsize=(10, 6))
        
        heir_means = []
        heir_stds = []
        neighbor_means = []
        neighbor_stds = []
        
        for year in self.years:
            heir_col = f'heir_ndvi_{year}'
            neighbor_col = f'neighbor_ndvi_{year}'
            
            if heir_col in analysis_df.columns and neighbor_col in analysis_df.columns:
                heir_means.append(analysis_df[heir_col].mean())
                heir_stds.append(analysis_df[heir_col].std())
                neighbor_means.append(analysis_df[neighbor_col].mean())
                neighbor_stds.append(analysis_df[neighbor_col].std())
        
        plt.errorbar(self.years, heir_means, yerr=heir_stds, 
                    label='Heirs Properties', marker='o', capsize=5)
        plt.errorbar(self.years, neighbor_means, yerr=neighbor_stds,
                    label='Non-heirs Properties', marker='s', capsize=5)
        
        plt.xlabel('Year')
        plt.ylabel('NDVI Value')
        plt.title('NDVI Time Series Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(self.output_dir / 'ndvi_time_series.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Distribution plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('NDVI Value Distributions by Year')
        
        for i, year in enumerate(self.years):
            heir_col = f'heir_ndvi_{year}'
            neighbor_col = f'neighbor_ndvi_{year}'
            
            if heir_col in analysis_df.columns and neighbor_col in analysis_df.columns:
                sns.kdeplot(data=analysis_df, x=heir_col, label='Heirs', ax=axes[i])
                sns.kdeplot(data=analysis_df, x=neighbor_col, label='Non-heirs', ax=axes[i])
                
                axes[i].set_title(year)
                axes[i].set_xlabel('NDVI Value')
                axes[i].set_ylabel('Density')
                axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'ndvi_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Difference boxplots
        differences = []
        
        for year in self.years:
            heir_col = f'heir_ndvi_{year}'
            neighbor_col = f'neighbor_ndvi_{year}'
            
            if heir_col in analysis_df.columns and neighbor_col in analysis_df.columns:
                diff = analysis_df[heir_col] - analysis_df[neighbor_col]
                differences.append(diff)
        
        plt.figure(figsize=(10, 6))
        plt.boxplot(differences, labels=self.years)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        plt.xlabel('Year')
        plt.ylabel('NDVI Difference (Heirs - Non-heirs)')
        plt.title('NDVI Differences Between Heirs and Non-heirs Properties')
        
        plt.savefig(self.output_dir / 'ndvi_differences.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Run statistical analysis as a standalone script."""
    try:
        # Load match and NDVI data
        matches_file = Path("output/matches/vance_matches.csv")
        heirs_ndvi_file = Path("output/ndvi/vance_heirs_ndvi.csv")
        neighbors_ndvi_file = Path("output/ndvi/vance_neighbors_ndvi.csv")
        
        # Check if files exist
        if not all(f.exists() for f in [matches_file, heirs_ndvi_file, neighbors_ndvi_file]):
            print("Please run property_matcher.py and ndvi_processor.py first")
            return
        
        # Load data
        print("\nLoading data...")
        matches_df = pd.read_csv(matches_file)
        heirs_ndvi = pd.read_csv(heirs_ndvi_file)
        neighbors_ndvi = pd.read_csv(neighbors_ndvi_file)
        
        print(f"Loaded:")
        print(f"- {len(matches_df)} property matches")
        print(f"- {len(heirs_ndvi)} heirs properties with NDVI")
        print(f"- {len(neighbors_ndvi)} neighbor properties with NDVI")
        
        # Initialize analyzer
        analyzer = StatsAnalyzer()
        
        # Calculate statistics
        print("\nCalculating statistics...")
        stats = analyzer.calculate_statistics(matches_df, heirs_ndvi)
        
        # Save statistics
        stats_file = analyzer.output_dir / "vance_statistics.txt"
        with open(stats_file, "w") as f:
            f.write("NDVI Analysis Statistics - Vance County\n")
            f.write("====================================\n\n")
            f.write(f"Sample size: {stats['sample_size']}\n")
            f.write(f"Years analyzed: {', '.join(stats['years_analyzed'])}\n\n")
            
            for year, year_stats in stats['comparisons'].items():
                f.write(f"\nYear {year}:\n")
                f.write("-" * (len(year) + 6) + "\n")
                f.write(f"Sample size: {year_stats['sample_size']}\n")
                f.write(f"Heirs properties: mean={year_stats['heirs_mean']:.4f}, std={year_stats['heirs_std']:.4f}\n")
                f.write(f"Non-heirs properties: mean={year_stats['neighbors_mean']:.4f}, std={year_stats['neighbors_std']:.4f}\n")
                f.write(f"Mean difference: {year_stats['mean_difference']:.4f} ± {year_stats['difference_std']:.4f}\n")
                f.write(f"T-statistic: {year_stats['t_statistic']:.4f}\n")
                f.write(f"P-value: {year_stats['p_value']:.4f}\n")
        
        print(f"\nSaved statistics to {stats_file}")
        
        # Create visualizations
        print("\nCreating visualizations...")
        analyzer.create_visualizations(matches_df, heirs_ndvi)
        
        print(f"\nAnalysis results saved to {analyzer.output_dir}/")
        
    except Exception as e:
        print(f"\nError during statistical analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 