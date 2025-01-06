import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from src.analysis.stats_analyzer import StatsAnalyzer

@pytest.fixture
def test_data():
    """Create test NDVI data."""
    np.random.seed(42)
    n_properties = 100
    
    # Create test data with known trends
    data = {
        'property_id': range(n_properties),
        'is_heir': [i < n_properties/2 for i in range(n_properties)],  # 50/50 split
        'ndvi_2018': np.random.normal(0.3, 0.1, n_properties),
        'ndvi_2020': np.random.normal(0.35, 0.1, n_properties),
        'ndvi_2022': np.random.normal(0.4, 0.1, n_properties),
    }
    
    # Add slightly different trends for heirs vs non-heirs
    df = pd.DataFrame(data)
    df.loc[df['is_heir'], 'ndvi_trend_slope'] = np.random.normal(0.05, 0.01, int(n_properties/2))
    df.loc[~df['is_heir'], 'ndvi_trend_slope'] = np.random.normal(0.03, 0.01, int(n_properties/2))
    
    return df

@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

def test_load_ndvi_data(test_data, temp_dir):
    """Test loading NDVI data."""
    # Save test data
    test_file = Path(temp_dir) / 'test_ndvi.parquet'
    test_data.to_parquet(test_file)
    
    # Test loading
    analyzer = StatsAnalyzer(temp_dir)
    loaded_data = analyzer.load_ndvi_data(test_file)
    
    assert len(loaded_data) == len(test_data)
    assert all(loaded_data.columns == test_data.columns)

def test_calculate_trend_statistics(test_data, temp_dir):
    """Test trend statistics calculation."""
    analyzer = StatsAnalyzer(temp_dir)
    stats_df = analyzer.calculate_trend_statistics(test_data)
    
    # Check basic statistics presence
    assert 'heirs' in stats_df.columns
    assert 'non_heirs' in stats_df.columns
    assert 't_statistic' in stats_df.index
    assert 'p_value' in stats_df.index
    
    # Verify statistics are reasonable
    assert stats_df.loc['mean', 'heirs'] > stats_df.loc['mean', 'non_heirs']  # Based on our test data
    assert not pd.isna(stats_df.loc['t_statistic', 'heirs'])
    assert not pd.isna(stats_df.loc['p_value', 'heirs'])

def test_plot_trend_comparison(test_data, temp_dir):
    """Test visualization creation."""
    analyzer = StatsAnalyzer(temp_dir)
    plot_path = Path(temp_dir) / 'test_plot.png'
    
    analyzer.plot_trend_comparison(test_data, str(plot_path))
    assert plot_path.exists()

def test_run_analysis(test_data, temp_dir):
    """Test complete analysis workflow."""
    # Save test data
    test_file = Path(temp_dir) / 'test_ndvi.parquet'
    test_data.to_parquet(test_file)
    
    # Run analysis
    analyzer = StatsAnalyzer(temp_dir)
    stats_df = analyzer.run_analysis(str(test_file))
    
    # Check outputs
    assert (Path(temp_dir) / 'ndvi_comparison_stats.csv').exists()
    assert (Path(temp_dir) / 'ndvi_trend_comparison.png').exists()
    assert isinstance(stats_df, pd.DataFrame)
    assert 'heirs' in stats_df.columns
    assert 'non_heirs' in stats_df.columns 