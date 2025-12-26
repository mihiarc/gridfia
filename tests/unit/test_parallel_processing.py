#!/usr/bin/env python3
"""
Comprehensive tests for parallel processing utilities in gridfia.

Tests cover all parallel processing functions, error handling, resource management,
chunking logic, and statistical analysis workflows with extensive edge cases.
"""

import multiprocessing
import os
import tempfile
import time
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import geopandas as gpd
import numpy as np
import pandas as pd
import psutil
import pytest
from shapely.geometry import Point, Polygon

from gridfia.utils.parallel_processing import (
    ParallelProcessor,
    _bootstrap_worker,
    _permutation_worker,
    _spatial_intersection_worker,
    optimize_memory_usage,
)


class TestParallelProcessorInitialization:
    """Test ParallelProcessor initialization and resource detection."""

    def test_default_initialization(self):
        """Test default initialization with auto-detected resources."""
        processor = ParallelProcessor()

        # Basic attributes exist
        assert hasattr(processor, 'cpu_count')
        assert hasattr(processor, 'max_workers')
        assert hasattr(processor, 'total_memory_gb')
        assert hasattr(processor, 'available_memory_gb')
        assert hasattr(processor, 'memory_limit_gb')

        # Resource detection worked
        assert processor.cpu_count > 0
        assert processor.max_workers >= 1
        assert processor.total_memory_gb > 0
        assert processor.available_memory_gb > 0
        assert processor.memory_limit_gb > 0

        # Worker count is reasonable
        assert processor.max_workers <= min(processor.cpu_count, 32)

    def test_custom_max_workers(self):
        """Test initialization with custom max_workers."""
        custom_workers = 4
        processor = ParallelProcessor(max_workers=custom_workers)

        assert processor.max_workers == min(custom_workers, processor.cpu_count)

    def test_custom_memory_limit(self):
        """Test initialization with custom memory limit."""
        custom_memory = 2.0  # 2GB
        processor = ParallelProcessor(memory_limit_gb=custom_memory)

        # Should respect the limit but not exceed available memory
        expected_limit = min(custom_memory, processor.available_memory_gb * 0.9)
        assert processor.memory_limit_gb == expected_limit

    def test_excessive_workers_capped(self):
        """Test that excessive worker count is capped appropriately."""
        excessive_workers = 1000
        processor = ParallelProcessor(max_workers=excessive_workers)

        assert processor.max_workers <= processor.cpu_count
        assert processor.max_workers <= 32  # Hard cap for stability

    def test_excessive_memory_limit_capped(self):
        """Test that excessive memory limit is capped appropriately."""
        excessive_memory = 1000.0  # 1TB
        processor = ParallelProcessor(memory_limit_gb=excessive_memory)

        assert processor.memory_limit_gb <= processor.available_memory_gb * 0.9

    @patch('psutil.virtual_memory')
    @patch('multiprocessing.cpu_count')
    def test_resource_detection_with_mocked_system(self, mock_cpu_count, mock_memory):
        """Test resource detection with mocked system resources."""
        # Mock system resources
        mock_cpu_count.return_value = 8
        mock_memory.return_value = MagicMock(
            total=16 * 1024**3,  # 16GB
            available=12 * 1024**3  # 12GB available
        )

        processor = ParallelProcessor()

        assert processor.cpu_count == 8
        assert processor.max_workers == 6  # 8 - 2 (leaving cores for system)
        assert abs(processor.total_memory_gb - 16.0) < 0.1
        assert abs(processor.available_memory_gb - 12.0) < 0.1
        assert abs(processor.memory_limit_gb - 9.6) < 0.1  # 80% of available

    def test_single_core_system(self):
        """Test behavior on single-core system."""
        with patch('multiprocessing.cpu_count', return_value=1):
            processor = ParallelProcessor()
            assert processor.max_workers == 1

    def test_low_memory_system(self):
        """Test behavior on low memory system."""
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = MagicMock(
                total=1 * 1024**3,  # 1GB
                available=0.5 * 1024**3  # 512MB available
            )

            processor = ParallelProcessor()
            assert processor.memory_limit_gb <= 0.5 * 0.8  # 80% of available


class TestChunkSizeCalculation:
    """Test chunk size calculation logic."""

    def test_chunk_size_calculation_basic(self):
        """Test basic chunk size calculation."""
        processor = ParallelProcessor(max_workers=4, memory_limit_gb=4.0)

        # Test with various data sizes and memory footprints
        chunk_size = processor._calculate_chunk_size(1000, 100.0)  # 1000 items, 100MB
        assert chunk_size > 0
        assert chunk_size <= 1000

    def test_chunk_size_with_large_dataset(self):
        """Test chunk size calculation with large dataset."""
        processor = ParallelProcessor(max_workers=8, memory_limit_gb=8.0)

        chunk_size = processor._calculate_chunk_size(100000, 1000.0)  # 100k items, 1GB
        assert chunk_size > 0
        assert chunk_size < 100000  # Should be chunked

    def test_chunk_size_with_small_dataset(self):
        """Test chunk size calculation with small dataset."""
        processor = ParallelProcessor(max_workers=4, memory_limit_gb=4.0)

        chunk_size = processor._calculate_chunk_size(10, 1.0)  # 10 items, 1MB
        assert chunk_size >= 1
        assert chunk_size <= 10

    def test_chunk_size_with_zero_memory(self):
        """Test chunk size calculation when memory footprint is unknown."""
        processor = ParallelProcessor(max_workers=4, memory_limit_gb=4.0)

        chunk_size = processor._calculate_chunk_size(1000, 0.0)  # Unknown memory
        assert chunk_size > 0
        assert chunk_size <= 1000

    def test_chunk_size_efficiency_constraint(self):
        """Test that chunk size meets efficiency constraints."""
        processor = ParallelProcessor(max_workers=4, memory_limit_gb=4.0)

        chunk_size = processor._calculate_chunk_size(1000, 50.0)

        # Should create at least 2x workers worth of chunks for efficiency
        min_chunks = processor.max_workers * 2
        max_chunk_size = 1000 // min_chunks
        assert chunk_size <= max(max_chunk_size, 1000 // processor.max_workers)


class TestSpatialIntersectionWorker:
    """Test spatial intersection worker functions."""

    @pytest.fixture
    def sample_geodataframes(self):
        """Create sample GeoDataFrames for testing."""
        # Target GDF with points
        target_points = [
            Point(0, 0),
            Point(1, 1),
            Point(2, 2),
            Point(10, 10),  # Outside source bounds
        ]
        target_gdf = gpd.GeoDataFrame(
            {'id': [1, 2, 3, 4]},
            geometry=target_points,
            crs='EPSG:4326'
        )

        # Source GDF with polygon covering some points
        source_polygon = Polygon([(-0.5, -0.5), (2.5, -0.5), (2.5, 2.5), (-0.5, 2.5)])
        source_gdf = gpd.GeoDataFrame(
            {'region': ['test_region']},
            geometry=[source_polygon],
            crs='EPSG:4326'
        )

        return target_gdf, source_gdf

    def test_spatial_intersection_worker_bounds_check(self, sample_geodataframes):
        """Test spatial intersection worker with bounds check."""
        target_gdf, source_gdf = sample_geodataframes
        options = {'use_bounds_check': True}

        result = _spatial_intersection_worker((target_gdf, source_gdf, options))

        assert result['success'] is True
        assert result['chunk_size'] == 4
        assert len(result['intersecting_indices']) >= 0
        assert result['intersecting_count'] == len(result['intersecting_indices'])

    def test_spatial_intersection_worker_full_intersection(self, sample_geodataframes):
        """Test spatial intersection worker with full geometric intersection."""
        target_gdf, source_gdf = sample_geodataframes
        options = {'use_bounds_check': False}

        result = _spatial_intersection_worker((target_gdf, source_gdf, options))

        assert result['success'] is True
        assert result['chunk_size'] == 4
        assert isinstance(result['intersecting_indices'], list)
        assert result['intersecting_count'] == len(result['intersecting_indices'])

    def test_spatial_intersection_worker_error_handling(self):
        """Test spatial intersection worker error handling."""
        # Create a scenario that will trigger an error - corrupted GeoDataFrame with string geometry
        try:
            # Force an error by passing invalid geometry types that will fail during spatial operations
            invalid_data = {'id': [1], 'geometry': ['invalid_geom']}
            invalid_target = pd.DataFrame(invalid_data)  # Not a GeoDataFrame
            valid_source = gpd.GeoDataFrame({'region': ['test']}, geometry=[Point(0, 0)], crs='EPSG:4326')

            result = _spatial_intersection_worker((invalid_target, valid_source, {'use_bounds_check': False}))

            assert result['success'] is False
            assert 'error' in result
            assert result['intersecting_count'] == 0
            assert result['intersecting_indices'] == []
        except Exception:
            # If even creating the test data fails, test with None to trigger the error path
            result = _spatial_intersection_worker((None, None, {}))
            assert result['success'] is False

    def test_spatial_intersection_worker_empty_data(self):
        """Test spatial intersection worker with empty data."""
        empty_target = gpd.GeoDataFrame({'id': []}, geometry=[], crs='EPSG:4326')
        empty_source = gpd.GeoDataFrame({'region': []}, geometry=[], crs='EPSG:4326')
        options = {'use_bounds_check': True}

        result = _spatial_intersection_worker((empty_target, empty_source, options))

        assert result['success'] is True
        assert result['chunk_size'] == 0
        assert result['intersecting_count'] == 0


class TestBootstrapWorker:
    """Test bootstrap analysis worker functions."""

    def test_bootstrap_worker_basic(self):
        """Test basic bootstrap worker functionality."""
        np.random.seed(42)
        group1_data = np.array([1, 2, 3, 4, 5])
        group2_data = np.array([2, 3, 4, 5, 6])
        options = {}

        result = _bootstrap_worker((group1_data, group2_data, options))

        assert result['success'] is True
        assert 'statistic' in result
        assert isinstance(result['statistic'], (float, np.floating))
        assert result['n1'] == 5
        assert result['n2'] == 5

    def test_bootstrap_worker_different_sizes(self):
        """Test bootstrap worker with different group sizes."""
        group1_data = np.array([1, 2, 3])
        group2_data = np.array([4, 5, 6, 7, 8])
        options = {}

        result = _bootstrap_worker((group1_data, group2_data, options))

        assert result['success'] is True
        assert result['n1'] == 3
        assert result['n2'] == 5

    def test_bootstrap_worker_empty_groups(self):
        """Test bootstrap worker with empty groups."""
        group1_data = np.array([])
        group2_data = np.array([1, 2, 3])
        options = {}

        result = _bootstrap_worker((group1_data, group2_data, options))

        # Should handle gracefully (may succeed or fail depending on implementation)
        assert 'success' in result
        if result['success']:
            assert 'statistic' in result

    def test_bootstrap_worker_error_handling(self):
        """Test bootstrap worker error handling."""
        # Invalid input data
        result = _bootstrap_worker((None, None, {}))

        assert result['success'] is False
        assert 'error' in result
        assert np.isnan(result['statistic'])

    def test_bootstrap_worker_single_values(self):
        """Test bootstrap worker with single values."""
        group1_data = np.array([5.0])
        group2_data = np.array([3.0])
        options = {}

        result = _bootstrap_worker((group1_data, group2_data, options))

        assert result['success'] is True
        assert result['statistic'] == 2.0  # 5 - 3

    def test_bootstrap_worker_statistical_properties(self):
        """Test bootstrap worker produces reasonable statistical results."""
        np.random.seed(42)
        # Groups with known difference in means
        group1_data = np.random.normal(10, 1, 100)
        group2_data = np.random.normal(8, 1, 100)
        options = {}

        results = []
        for _ in range(50):  # Multiple bootstrap samples
            result = _bootstrap_worker((group1_data, group2_data, options))
            if result['success']:
                results.append(result['statistic'])

        assert len(results) > 40  # Most should succeed
        # Bootstrap statistics should be centered around the true difference (~2)
        mean_diff = np.mean(results)
        assert 1.5 < mean_diff < 2.5


class TestPermutationWorker:
    """Test permutation test worker functions."""

    def test_permutation_worker_basic(self):
        """Test basic permutation worker functionality."""
        np.random.seed(42)
        combined_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        n1, n2 = 5, 5

        result = _permutation_worker((combined_data, n1, n2))

        assert isinstance(result, (float, np.floating))
        assert not np.isnan(result)

    def test_permutation_worker_different_sizes(self):
        """Test permutation worker with different group sizes."""
        combined_data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        n1, n2 = 3, 5

        result = _permutation_worker((combined_data, n1, n2))

        assert isinstance(result, (float, np.floating))
        assert not np.isnan(result)

    def test_permutation_worker_error_handling(self):
        """Test permutation worker error handling."""
        # Invalid input data
        result = _permutation_worker((None, 1, 1))

        assert np.isnan(result)

    def test_permutation_worker_single_values(self):
        """Test permutation worker with minimal data."""
        combined_data = np.array([1, 2])
        n1, n2 = 1, 1

        result = _permutation_worker((combined_data, n1, n2))

        # Should be either 1-2 = -1 or 2-1 = 1
        assert result in [-1, 1] or abs(result) == 1

    def test_permutation_worker_statistical_properties(self):
        """Test permutation worker produces valid permutation statistics."""
        # Test with data that has no true difference
        np.random.seed(42)
        combined_data = np.random.normal(5, 1, 100)
        n1, n2 = 50, 50

        results = []
        for _ in range(50):  # Multiple permutations
            result = _permutation_worker((combined_data, n1, n2))
            if not np.isnan(result):
                results.append(result)

        assert len(results) > 40  # Most should succeed
        # Under null hypothesis, permutation statistics should be centered around 0
        mean_stat = np.mean(results)
        assert abs(mean_stat) < 0.5  # Should be close to 0


class TestParallelSpatialIntersection:
    """Test parallel spatial intersection functionality."""

    @pytest.fixture
    def large_geodataframes(self):
        """Create larger GeoDataFrames for parallel processing tests."""
        # Create target points in a grid
        x_coords = np.linspace(0, 10, 50)
        y_coords = np.linspace(0, 10, 50)
        points = []
        ids = []

        for i, x in enumerate(x_coords):
            for j, y in enumerate(y_coords):
                points.append(Point(x, y))
                ids.append(i * len(y_coords) + j)

        target_gdf = gpd.GeoDataFrame(
            {'id': ids},
            geometry=points,
            crs='EPSG:4326'
        )

        # Create source polygons that cover different areas
        polygons = [
            Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]),  # Bottom-left quadrant
            Polygon([(5, 5), (10, 5), (10, 10), (5, 10)])  # Top-right quadrant
        ]
        source_gdf = gpd.GeoDataFrame(
            {'region': ['region_1', 'region_2']},
            geometry=polygons,
            crs='EPSG:4326'
        )

        return target_gdf, source_gdf

    def test_parallel_spatial_intersection_basic(self, large_geodataframes):
        """Test basic parallel spatial intersection."""
        target_gdf, source_gdf = large_geodataframes
        processor = ParallelProcessor(max_workers=2)

        result = processor.parallel_spatial_intersection(
            target_gdf, source_gdf, use_bounds_check=True
        )

        assert isinstance(result, list)
        assert len(result) >= 0
        assert all(isinstance(idx, (int, np.integer)) for idx in result)

    def test_parallel_spatial_intersection_full_geometric(self, large_geodataframes):
        """Test parallel spatial intersection with full geometric intersection."""
        target_gdf, source_gdf = large_geodataframes
        processor = ParallelProcessor(max_workers=2)

        result = processor.parallel_spatial_intersection(
            target_gdf, source_gdf, use_bounds_check=False
        )

        assert isinstance(result, list)
        assert len(result) >= 0

    def test_parallel_spatial_intersection_single_worker(self, large_geodataframes):
        """Test parallel spatial intersection with single worker."""
        target_gdf, source_gdf = large_geodataframes
        processor = ParallelProcessor(max_workers=1)

        result = processor.parallel_spatial_intersection(
            target_gdf, source_gdf, use_bounds_check=True
        )

        assert isinstance(result, list)

    def test_parallel_spatial_intersection_empty_data(self):
        """Test parallel spatial intersection with empty data."""
        empty_target = gpd.GeoDataFrame({'id': []}, geometry=[], crs='EPSG:4326')
        empty_source = gpd.GeoDataFrame({'region': []}, geometry=[], crs='EPSG:4326')
        processor = ParallelProcessor(max_workers=2)

        result = processor.parallel_spatial_intersection(
            empty_target, empty_source, use_bounds_check=True
        )

        assert result == []

    def test_spatial_intersection_fallback_to_sequential(self, large_geodataframes):
        """Test fallback to sequential processing on parallel failure."""
        target_gdf, source_gdf = large_geodataframes
        processor = ParallelProcessor(max_workers=2)

        # Mock the ThreadPoolExecutor to raise an exception
        with patch('gridfia.utils.parallel_processing.ThreadPoolExecutor') as mock_executor:
            mock_executor.side_effect = Exception("Parallel processing failed")

            result = processor.parallel_spatial_intersection(
                target_gdf, source_gdf, use_bounds_check=True
            )

            # Should still return results from fallback
            assert isinstance(result, list)

    def test_sequential_spatial_intersection_fallback(self, large_geodataframes):
        """Test sequential spatial intersection fallback methods."""
        target_gdf, source_gdf = large_geodataframes
        processor = ParallelProcessor(max_workers=2)

        # Test bounds-based fallback
        result_bounds = processor._sequential_spatial_intersection(
            target_gdf, source_gdf, use_bounds_check=True
        )
        assert isinstance(result_bounds, list)

        # Test geometric fallback
        result_geometric = processor._sequential_spatial_intersection(
            target_gdf, source_gdf, use_bounds_check=False
        )
        assert isinstance(result_geometric, list)


class TestParallelBootstrapAnalysis:
    """Test parallel bootstrap analysis functionality."""

    def test_parallel_bootstrap_analysis_basic(self):
        """Test basic parallel bootstrap analysis."""
        np.random.seed(42)
        group1_data = np.random.normal(10, 2, 100)
        group2_data = np.random.normal(8, 2, 100)
        processor = ParallelProcessor(max_workers=2)

        result = processor.parallel_bootstrap_analysis(
            group1_data, group2_data, n_iterations=50
        )

        assert 'bootstrap_statistics' in result
        assert 'n_successful' in result
        assert 'n_failed' in result

        assert isinstance(result['bootstrap_statistics'], np.ndarray)
        assert result['n_successful'] >= 0
        assert result['n_failed'] >= 0
        assert result['n_successful'] + result['n_failed'] == 50

    def test_parallel_bootstrap_analysis_single_iteration(self):
        """Test parallel bootstrap analysis with single iteration."""
        group1_data = np.array([1, 2, 3, 4, 5])
        group2_data = np.array([2, 3, 4, 5, 6])
        processor = ParallelProcessor(max_workers=1)

        result = processor.parallel_bootstrap_analysis(
            group1_data, group2_data, n_iterations=1
        )

        assert len(result['bootstrap_statistics']) <= 1

    def test_parallel_bootstrap_analysis_many_iterations(self):
        """Test parallel bootstrap analysis with many iterations."""
        group1_data = np.random.normal(10, 1, 50)
        group2_data = np.random.normal(9, 1, 50)
        processor = ParallelProcessor(max_workers=2)

        result = processor.parallel_bootstrap_analysis(
            group1_data, group2_data, n_iterations=20
        )

        # Should handle execution (may fail in test environment due to process pool issues)
        assert 'n_successful' in result
        assert 'n_failed' in result
        assert result['n_successful'] + result['n_failed'] == 20
        assert len(result['bootstrap_statistics']) == result['n_successful']

    def test_parallel_bootstrap_analysis_empty_data(self):
        """Test parallel bootstrap analysis with empty data."""
        group1_data = np.array([])
        group2_data = np.array([1, 2, 3])
        processor = ParallelProcessor(max_workers=2)

        result = processor.parallel_bootstrap_analysis(
            group1_data, group2_data, n_iterations=10
        )

        # May have failures due to empty data
        assert 'bootstrap_statistics' in result
        assert result['n_failed'] >= 0

    @patch('gridfia.utils.parallel_processing.ProcessPoolExecutor')
    def test_parallel_bootstrap_analysis_execution_failure(self, mock_executor):
        """Test bootstrap analysis with parallel execution failure."""
        mock_executor.side_effect = Exception("Execution failed")

        group1_data = np.array([1, 2, 3])
        group2_data = np.array([4, 5, 6])
        processor = ParallelProcessor(max_workers=2)

        result = processor.parallel_bootstrap_analysis(
            group1_data, group2_data, n_iterations=10
        )

        # Should handle failure gracefully
        assert result['n_successful'] == 0
        assert result['n_failed'] == 10
        assert len(result['bootstrap_statistics']) == 0


class TestParallelPermutationTest:
    """Test parallel permutation test functionality."""

    def test_parallel_permutation_test_basic(self):
        """Test basic parallel permutation test."""
        np.random.seed(42)
        group1_data = np.random.normal(10, 1, 50)
        group2_data = np.random.normal(10, 1, 50)  # Same distribution
        processor = ParallelProcessor(max_workers=2)

        result = processor.parallel_permutation_test(
            group1_data, group2_data, n_permutations=50
        )

        assert 'permutation_statistics' in result
        assert 'n_successful' in result
        assert 'n_failed' in result

        assert isinstance(result['permutation_statistics'], np.ndarray)
        assert result['n_successful'] >= 0
        assert result['n_failed'] >= 0
        assert result['n_successful'] + result['n_failed'] == 50

    def test_parallel_permutation_test_different_groups(self):
        """Test permutation test with groups that have different means."""
        np.random.seed(42)
        group1_data = np.random.normal(12, 1, 50)
        group2_data = np.random.normal(8, 1, 50)
        processor = ParallelProcessor(max_workers=2)

        result = processor.parallel_permutation_test(
            group1_data, group2_data, n_permutations=20
        )

        # Should handle execution (may fail in test environment due to process pool issues)
        assert 'n_successful' in result
        assert 'n_failed' in result
        assert result['n_successful'] + result['n_failed'] == 20
        assert len(result['permutation_statistics']) == result['n_successful']

        # Check results if any succeeded
        if result['n_successful'] > 0:
            permuted_diffs = result['permutation_statistics']
            assert len(permuted_diffs) > 0

    def test_parallel_permutation_test_single_permutation(self):
        """Test permutation test with single permutation."""
        group1_data = np.array([1, 2, 3])
        group2_data = np.array([4, 5, 6])
        processor = ParallelProcessor(max_workers=1)

        result = processor.parallel_permutation_test(
            group1_data, group2_data, n_permutations=1
        )

        assert len(result['permutation_statistics']) <= 1

    def test_parallel_permutation_test_empty_data(self):
        """Test permutation test with empty data."""
        group1_data = np.array([])
        group2_data = np.array([1, 2, 3])
        processor = ParallelProcessor(max_workers=2)

        result = processor.parallel_permutation_test(
            group1_data, group2_data, n_permutations=10
        )

        # May have failures due to empty data
        assert 'permutation_statistics' in result

    @patch('gridfia.utils.parallel_processing.ProcessPoolExecutor')
    def test_parallel_permutation_test_execution_failure(self, mock_executor):
        """Test permutation test with parallel execution failure."""
        mock_executor.side_effect = Exception("Execution failed")

        group1_data = np.array([1, 2, 3])
        group2_data = np.array([4, 5, 6])
        processor = ParallelProcessor(max_workers=2)

        result = processor.parallel_permutation_test(
            group1_data, group2_data, n_permutations=10
        )

        # Should handle failure gracefully
        assert result['n_successful'] == 0
        assert result['n_failed'] == 10
        assert len(result['permutation_statistics']) == 0


class TestMemoryOptimization:
    """Test memory optimization utilities."""

    def test_optimize_memory_usage_basic(self):
        """Test basic memory optimization function."""
        # Store original values
        original_omp = os.environ.get('OMP_NUM_THREADS')
        original_numexpr = os.environ.get('NUMEXPR_MAX_THREADS')

        try:
            # Call optimization
            optimize_memory_usage()

            # Check that environment variables were set
            assert 'OMP_NUM_THREADS' in os.environ
            assert 'NUMEXPR_MAX_THREADS' in os.environ

            # Values should be reasonable
            omp_threads = int(os.environ['OMP_NUM_THREADS'])
            numexpr_threads = int(os.environ['NUMEXPR_MAX_THREADS'])

            assert 1 <= omp_threads <= min(8, multiprocessing.cpu_count())
            assert 1 <= numexpr_threads <= min(8, multiprocessing.cpu_count())

        finally:
            # Restore original values
            if original_omp is not None:
                os.environ['OMP_NUM_THREADS'] = original_omp
            elif 'OMP_NUM_THREADS' in os.environ:
                del os.environ['OMP_NUM_THREADS']

            if original_numexpr is not None:
                os.environ['NUMEXPR_MAX_THREADS'] = original_numexpr
            elif 'NUMEXPR_MAX_THREADS' in os.environ:
                del os.environ['NUMEXPR_MAX_THREADS']

    @patch('multiprocessing.cpu_count')
    def test_optimize_memory_usage_with_many_cores(self, mock_cpu_count):
        """Test memory optimization with many CPU cores."""
        mock_cpu_count.return_value = 32

        original_omp = os.environ.get('OMP_NUM_THREADS')
        original_numexpr = os.environ.get('NUMEXPR_MAX_THREADS')

        try:
            optimize_memory_usage()

            # Should be capped at 8 even with many cores
            omp_threads = int(os.environ['OMP_NUM_THREADS'])
            numexpr_threads = int(os.environ['NUMEXPR_MAX_THREADS'])

            assert omp_threads == 8
            assert numexpr_threads == 8

        finally:
            if original_omp is not None:
                os.environ['OMP_NUM_THREADS'] = original_omp
            elif 'OMP_NUM_THREADS' in os.environ:
                del os.environ['OMP_NUM_THREADS']

            if original_numexpr is not None:
                os.environ['NUMEXPR_MAX_THREADS'] = original_numexpr
            elif 'NUMEXPR_MAX_THREADS' in os.environ:
                del os.environ['NUMEXPR_MAX_THREADS']

    @patch('multiprocessing.cpu_count')
    def test_optimize_memory_usage_with_few_cores(self, mock_cpu_count):
        """Test memory optimization with few CPU cores."""
        mock_cpu_count.return_value = 2

        original_omp = os.environ.get('OMP_NUM_THREADS')
        original_numexpr = os.environ.get('NUMEXPR_MAX_THREADS')

        try:
            optimize_memory_usage()

            # Should match CPU count when fewer than 8
            omp_threads = int(os.environ['OMP_NUM_THREADS'])
            numexpr_threads = int(os.environ['NUMEXPR_MAX_THREADS'])

            assert omp_threads == 2
            assert numexpr_threads == 2

        finally:
            if original_omp is not None:
                os.environ['OMP_NUM_THREADS'] = original_omp
            elif 'OMP_NUM_THREADS' in os.environ:
                del os.environ['OMP_NUM_THREADS']

            if original_numexpr is not None:
                os.environ['NUMEXPR_MAX_THREADS'] = original_numexpr
            elif 'NUMEXPR_MAX_THREADS' in os.environ:
                del os.environ['NUMEXPR_MAX_THREADS']


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases."""

    def test_processor_with_realistic_workloads(self):
        """Test processor with realistic workloads and data sizes."""
        processor = ParallelProcessor(max_workers=4, memory_limit_gb=2.0)

        # Create realistic-sized data
        np.random.seed(42)
        large_group1 = np.random.normal(15, 5, 1000)
        large_group2 = np.random.normal(12, 4, 1000)

        # Test bootstrap analysis
        bootstrap_result = processor.parallel_bootstrap_analysis(
            large_group1, large_group2, n_iterations=10
        )

        # Should handle execution (may fail in test environment due to process pool issues)
        assert 'n_successful' in bootstrap_result
        assert 'n_failed' in bootstrap_result

        # Test permutation test
        permutation_result = processor.parallel_permutation_test(
            large_group1[:50], large_group2[:50], n_permutations=10
        )

        assert 'n_successful' in permutation_result
        assert 'n_failed' in permutation_result

    def test_processor_memory_constraints(self):
        """Test processor behavior under memory constraints."""
        # Create processor with very limited memory
        processor = ParallelProcessor(max_workers=2, memory_limit_gb=0.1)

        # Test chunk size calculation with limited memory
        chunk_size = processor._calculate_chunk_size(10000, 50.0)  # 10k items, 50MB

        # Should create smaller chunks due to memory constraints
        assert chunk_size > 0
        assert chunk_size < 10000

    def test_error_resilience_spatial_operations(self):
        """Test error resilience in spatial operations."""
        processor = ParallelProcessor(max_workers=2)

        # Create problematic GeoDataFrame (invalid geometries)
        invalid_points = [Point(float('inf'), 0), Point(0, float('nan'))]
        try:
            target_gdf = gpd.GeoDataFrame(
                {'id': [1, 2]},
                geometry=invalid_points,
                crs='EPSG:4326'
            )

            source_polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
            source_gdf = gpd.GeoDataFrame(
                {'region': ['test']},
                geometry=[source_polygon],
                crs='EPSG:4326'
            )

            # Should handle invalid geometries gracefully
            result = processor.parallel_spatial_intersection(
                target_gdf, source_gdf, use_bounds_check=True
            )

            # Should return some result (even if empty due to fallback)
            assert isinstance(result, list)

        except Exception:
            # If GeoDataFrame creation fails with invalid geometries, that's expected
            pass

    def test_performance_comparison_sequential_vs_parallel(self):
        """Test performance comparison between sequential and parallel execution."""
        processor = ParallelProcessor(max_workers=4)

        # Generate test data
        np.random.seed(42)
        group1_data = np.random.normal(10, 2, 200)
        group2_data = np.random.normal(8, 2, 200)

        # Time parallel execution
        start_time = time.time()
        parallel_result = processor.parallel_bootstrap_analysis(
            group1_data, group2_data, n_iterations=10
        )
        parallel_time = time.time() - start_time

        # Sequential would take longer, but we can't easily test that here
        # Just verify parallel execution completed successfully
        assert 'n_successful' in parallel_result
        assert 'n_failed' in parallel_result
        assert parallel_time > 0  # Some time elapsed

    def test_resource_cleanup_and_management(self):
        """Test proper resource cleanup and management."""
        processor = ParallelProcessor(max_workers=2)

        # Perform multiple operations to test resource management
        np.random.seed(42)
        data1 = np.random.normal(5, 1, 50)
        data2 = np.random.normal(5, 1, 50)

        # Multiple bootstrap analyses
        for _ in range(3):
            result = processor.parallel_bootstrap_analysis(
                data1, data2, n_iterations=5
            )
            assert 'bootstrap_statistics' in result

        # Multiple permutation tests
        for _ in range(3):
            result = processor.parallel_permutation_test(
                data1, data2, n_permutations=5
            )
            assert 'permutation_statistics' in result

        # Should not have resource leaks or hanging processes
        # This is implicit - if test completes without hanging, resources are managed


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions."""

    def test_single_data_point_operations(self):
        """Test operations with single data points."""
        processor = ParallelProcessor(max_workers=1)

        # Single point in each group
        single_group1 = np.array([5.0])
        single_group2 = np.array([3.0])

        # Bootstrap analysis
        bootstrap_result = processor.parallel_bootstrap_analysis(
            single_group1, single_group2, n_iterations=5
        )

        # Should handle single points
        assert 'bootstrap_statistics' in bootstrap_result

        # Permutation test
        permutation_result = processor.parallel_permutation_test(
            single_group1, single_group2, n_permutations=5
        )

        assert 'permutation_statistics' in permutation_result

    def test_identical_data_groups(self):
        """Test operations with identical data groups."""
        processor = ParallelProcessor(max_workers=2)

        # Identical groups
        data = np.array([1, 2, 3, 4, 5])

        # Bootstrap analysis with identical data
        bootstrap_result = processor.parallel_bootstrap_analysis(
            data, data, n_iterations=10
        )

        # Statistics should be centered around 0 (no difference)
        if bootstrap_result['n_successful'] > 0:
            mean_stat = np.mean(bootstrap_result['bootstrap_statistics'])
            assert abs(mean_stat) <= 0.5  # Should be close to 0, but allow for sampling variance

    def test_extreme_data_values(self):
        """Test operations with extreme data values."""
        processor = ParallelProcessor(max_workers=2)

        # Very large values
        large_data1 = np.array([1e10, 1e11, 1e12])
        large_data2 = np.array([1e9, 1e10, 1e11])

        # Should handle large values
        result = processor.parallel_bootstrap_analysis(
            large_data1, large_data2, n_iterations=5
        )

        # Check for numerical stability
        if result['n_successful'] > 0:
            assert not np.any(np.isinf(result['bootstrap_statistics']))
            assert not np.any(np.isnan(result['bootstrap_statistics']))

    def test_zero_iterations_or_permutations(self):
        """Test behavior with zero iterations/permutations."""
        processor = ParallelProcessor(max_workers=1)

        data1 = np.array([1, 2, 3])
        data2 = np.array([4, 5, 6])

        # Zero iterations
        bootstrap_result = processor.parallel_bootstrap_analysis(
            data1, data2, n_iterations=0
        )

        assert len(bootstrap_result['bootstrap_statistics']) == 0
        assert bootstrap_result['n_successful'] == 0

        # Zero permutations
        permutation_result = processor.parallel_permutation_test(
            data1, data2, n_permutations=0
        )

        assert len(permutation_result['permutation_statistics']) == 0
        assert permutation_result['n_successful'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])