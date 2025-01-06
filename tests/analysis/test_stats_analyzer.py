import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.analysis.stats_analyzer import StatsAnalyzer

def test_stats_analyzer_init(tmp_path):
    """Test StatsAnalyzer initialization."""
    analyzer = StatsAnalyzer(output_dir=str(tmp_path))
    assert analyzer.years == ['2018', '2020', '2022']
    assert analyzer.output_dir == Path(tmp_path)
    assert analyzer.output_dir.exists()

def test_calculate_statistics(sample_matches, sample_ndvi_data):
    """Test statistics calculation."""
    analyzer = StatsAnalyzer()
    stats = analyzer.calculate_statistics(sample_matches, sample_ndvi_data)
    
    # Verify basic structure
    assert 'sample_size' in stats
    assert 'years_analyzed' in stats
    assert 'comparisons' in stats
    
    # Verify year-specific statistics
    for year in analyzer.years:
        if year in stats['comparisons']:
            year_stats = stats['comparisons'][year]
            assert 'sample_size' in year_stats
            assert 'heirs_mean' in year_stats
            assert 'heirs_std' in year_stats
            assert 'neighbors_mean' in year_stats
            assert 'neighbors_std' in year_stats
            assert 'mean_difference' in year_stats
            assert 'difference_std' in year_stats
            assert 't_statistic' in year_stats
            assert 'p_value' in year_stats

def test_create_visualizations(tmp_path, sample_matches, sample_ndvi_data):
    """Test visualization creation."""
    analyzer = StatsAnalyzer(output_dir=str(tmp_path))
    analyzer.create_visualizations(sample_matches, sample_ndvi_data)
    
    # Verify output files
    assert (analyzer.output_dir / 'ndvi_time_series.png').exists()
    assert (analyzer.output_dir / 'analysis_data.csv').exists()
    assert (analyzer.output_dir / 'time_series_data.csv').exists()
    assert (analyzer.output_dir / 'differences_data.csv').exists()

def test_empty_data_handling():
    """Test handling of empty data."""
    analyzer = StatsAnalyzer()
    empty_matches = pd.DataFrame(columns=['heir_id', 'neighbor_id'])
    empty_ndvi = pd.DataFrame(columns=['property_id', 'property_type', 'ndvi_2018', 'ndvi_2020', 'ndvi_2022'])
    
    # Should handle empty data gracefully
    stats = analyzer.calculate_statistics(empty_matches, empty_ndvi)
    assert stats['sample_size'] == 0
    assert len(stats['comparisons']) == 0
    assert stats['years_analyzed'] == analyzer.years

def test_invalid_data_handling():
    """Test handling of invalid data."""
    analyzer = StatsAnalyzer()
    
    # Create invalid data
    invalid_matches = pd.DataFrame({
        'heir_id': [1],
        'neighbor_id': [2]
    })
    invalid_ndvi = pd.DataFrame({
        'property_id': [1],
        'property_type': ['heir'],
        'ndvi_2020': ['invalid']
    })
    
    # Should raise ValueError for invalid NDVI values
    with pytest.raises(ValueError, match="Invalid NDVI values"):
        analyzer.calculate_statistics(invalid_matches, invalid_ndvi)
    
    # Test missing columns
    missing_columns_ndvi = pd.DataFrame({
        'property_id': [1],
        'ndvi_2020': [0.5]
    })
    
    with pytest.raises(KeyError, match="Missing required columns"):
        analyzer.calculate_statistics(invalid_matches, missing_columns_ndvi)
    
    # Test out of range NDVI values
    out_of_range_ndvi = pd.DataFrame({
        'property_id': [1],
        'property_type': ['heir'],
        'ndvi_2020': [2.0]  # NDVI should be between -1 and 1
    })
    
    with pytest.raises(ValueError, match="NDVI values out of range"):
        analyzer.calculate_statistics(invalid_matches, out_of_range_ndvi) 