#!/usr/bin/env python3
"""
Parallel Processing Utilities for GridFIA

This module provides parallel processing capabilities optimized for
multi-core systems to accelerate spatial sampling, species extraction,
and statistical computations.

.. warning::
   This module is experimental and not yet integrated into the main
   processing pipeline. The implementation is complete but needs
   thorough testing and optimization for production use.

.. todo::
   Integration and optimization tasks:
   
   - [ ] Integrate with ForestMetricsProcessor for parallel calculations
   - [ ] Add unit tests for all parallel methods
   - [ ] Benchmark performance vs sequential processing
   - [ ] Optimize chunk size calculation for different data types
   - [ ] Add GPU acceleration support (CuPy/Rapids)
   - [ ] Implement adaptive worker pool sizing
   - [ ] Add progress bars for long-running operations
   - [ ] Create usage examples and documentation
   
   Target Version: v0.4.0
   Priority: Low (performance optimization)
   Dependencies: Core pipeline must be stable first

Example Usage::

    from gridfia.utils import ParallelProcessor
    
    processor = ParallelProcessor(max_workers=8)
    
    # Parallel spatial intersection
    intersecting = processor.parallel_spatial_intersection(
        target_gdf=parcels,
        source_gdf=reference_areas,
        use_bounds_check=True
    )
    
    # Parallel bootstrap analysis
    results = processor.parallel_bootstrap_analysis(
        group1_data=data1,
        group2_data=data2,
        n_iterations=10000
    )
"""

import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm
import geopandas as gpd

logger = logging.getLogger(__name__)

# Module-level worker functions for multiprocessing (must be pickleable)

def _spatial_intersection_worker(chunk_data: Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, Dict]) -> Dict[str, Any]:
    """
    Worker function for parallel spatial intersection.
    Must be at module level to be pickleable.
    """
    try:
        target_chunk, source_gdf, options = chunk_data
        
        # Perform spatial intersection
        if options.get('use_bounds_check', False):
            # Simple bounds-based intersection for performance
            source_bounds = source_gdf.total_bounds
            intersecting = []
            
            for idx, row in target_chunk.iterrows():
                geom_bounds = row.geometry.bounds
                if (geom_bounds[0] < source_bounds[2] and geom_bounds[2] > source_bounds[0] and
                    geom_bounds[1] < source_bounds[3] and geom_bounds[3] > source_bounds[1]):
                    intersecting.append(idx)
        else:
            # Full spatial intersection
            intersecting_gdf = gpd.sjoin(target_chunk, source_gdf, how='inner', predicate='intersects')
            intersecting = intersecting_gdf.index.unique().tolist()
        
        return {
            'success': True,
            'chunk_size': len(target_chunk),
            'intersecting_indices': intersecting,
            'intersecting_count': len(intersecting)
        }
        
    except Exception as e:
        logger.error(f"Spatial intersection worker error: {e}")
        return {
            'success': False,
            'error': str(e),
            'chunk_size': len(chunk_data[0]) if chunk_data else 0,
            'intersecting_indices': [],
            'intersecting_count': 0
        }


def _bootstrap_worker(data_tuple: Tuple[np.ndarray, np.ndarray, Dict]) -> Dict[str, Any]:
    """
    Worker function for bootstrap analysis.
    Must be at module level to be pickleable.
    """
    try:
        group1_data, group2_data, options = data_tuple
        
        # Resample with replacement
        np.random.seed()  # Ensure different seeds for each worker
        
        n1, n2 = len(group1_data), len(group2_data)
        resampled1 = np.random.choice(group1_data, size=n1, replace=True)
        resampled2 = np.random.choice(group2_data, size=n2, replace=True)
        
        # Calculate test statistic (difference in means)
        stat = np.mean(resampled1) - np.mean(resampled2)
        
        return {
            'success': True,
            'statistic': stat,
            'n1': n1,
            'n2': n2
        }
        
    except Exception as e:
        logger.error(f"Bootstrap worker error: {e}")
        return {
            'success': False,
            'error': str(e),
            'statistic': np.nan
        }

def _permutation_worker(data_tuple: Tuple[np.ndarray, int, int]) -> float:
    """
    Worker function for permutation test.
    Must be at module level to be pickleable.
    """
    try:
        combined_data, n1, n2 = data_tuple
        
        # Shuffle and split
        np.random.seed()  # Ensure different seeds for each worker
        shuffled = np.random.permutation(combined_data)
        
        group1_perm = shuffled[:n1]
        group2_perm = shuffled[n1:n1+n2]
        
        # Calculate test statistic
        return np.mean(group1_perm) - np.mean(group2_perm)
        
    except Exception as e:
        logger.error(f"Permutation worker error: {e}")
        return np.nan

class ParallelProcessor:
    """
    Handles parallel processing for GridFIA operations with automatic resource optimization.
    """
    
    def __init__(self, max_workers: Optional[int] = None, memory_limit_gb: Optional[float] = None):
        """
        Initialize parallel processor with automatic resource detection.
        
        Args:
            max_workers: Maximum number of worker processes (auto-detected if None)
            memory_limit_gb: Memory limit in GB (auto-detected if None)
        """
        # System resource detection
        self.cpu_count = multiprocessing.cpu_count()
        self.total_memory_gb = psutil.virtual_memory().total / (1024**3)
        self.available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Set optimal worker count (leave some cores for system)
        if max_workers is None:
            self.max_workers = max(1, min(self.cpu_count - 2, 32))  # Cap at 32 for stability
        else:
            self.max_workers = min(max_workers, self.cpu_count)
        
        # Set memory limit (use 80% of available memory)
        if memory_limit_gb is None:
            self.memory_limit_gb = self.available_memory_gb * 0.8
        else:
            self.memory_limit_gb = min(memory_limit_gb, self.available_memory_gb * 0.9)
        
        logger.info(f"ParallelProcessor initialized:")
        logger.info(f"  CPU cores: {self.cpu_count}, Workers: {self.max_workers}")
        logger.info(f"  Total memory: {self.total_memory_gb:.1f}GB, Limit: {self.memory_limit_gb:.1f}GB")
    
    def _calculate_chunk_size(self, data_size: int, data_memory_mb: float) -> int:
        """Calculate optimal chunk size based on data size and memory constraints."""
        # Target memory per chunk (MB)
        target_memory_per_chunk = (self.memory_limit_gb * 1024) / (self.max_workers * 2)
        
        # Calculate chunk size based on memory
        if data_memory_mb > 0:
            memory_based_chunk = max(1, int((target_memory_per_chunk / data_memory_mb) * data_size))
        else:
            memory_based_chunk = data_size
        
        # Ensure minimum efficiency (at least 2x workers worth of chunks)
        min_chunk_size = max(1, data_size // (self.max_workers * 4))
        max_chunk_size = max(min_chunk_size, data_size // self.max_workers)
        
        optimal_chunk_size = max(min_chunk_size, min(memory_based_chunk, max_chunk_size))
        
        logger.debug(f"Chunk size calculation: data_size={data_size}, "
                    f"memory_mb={data_memory_mb:.1f}, chunk_size={optimal_chunk_size}")
        
        return optimal_chunk_size

    def parallel_spatial_intersection(
        self, 
        target_gdf: gpd.GeoDataFrame, 
        source_gdf: gpd.GeoDataFrame,
        use_bounds_check: bool = False
    ) -> List[int]:
        """
        Perform spatial intersection using parallel processing.
        
        Args:
            target_gdf: Target GeoDataFrame to find intersections in
            source_gdf: Source GeoDataFrame to intersect with
            use_bounds_check: Use faster bounds-based intersection instead of full spatial
            
        Returns:
            List of indices from target_gdf that intersect with source_gdf
        """
        logger.info(f"Starting parallel spatial intersection: {len(target_gdf)} x {len(source_gdf)} parcels")
        
        # Estimate memory usage
        target_memory_mb = target_gdf.memory_usage(deep=True).sum() / (1024**2)
        source_memory_mb = source_gdf.memory_usage(deep=True).sum() / (1024**2)
        
        # Calculate optimal chunk size
        chunk_size = self._calculate_chunk_size(len(target_gdf), target_memory_mb)
        
        # Prepare chunks with metadata
        chunks = []
        options = {'use_bounds_check': use_bounds_check}
        
        for i in range(0, len(target_gdf), chunk_size):
            chunk = target_gdf.iloc[i:i+chunk_size].copy()
            chunks.append((chunk, source_gdf, options))
        
        logger.info(f"Processing {len(chunks)} chunks with {self.max_workers} workers")
        
        # Process chunks in parallel
        start_time = time.time()
        intersecting_indices = []
        
        try:
            # Use ThreadPoolExecutor for I/O-bound spatial operations to avoid serialization issues
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(_spatial_intersection_worker, chunks))
            
            # Collect results
            failed_chunks = 0
            for result in results:
                if result['success']:
                    intersecting_indices.extend(result['intersecting_indices'])
                else:
                    failed_chunks += 1
                    logger.warning(f"Chunk failed: {result.get('error', 'Unknown error')}")
            
            duration = time.time() - start_time
            logger.info(f"Spatial intersection completed in {duration:.2f}s")
            logger.info(f"Found {len(intersecting_indices)} intersecting parcels")
            
            if failed_chunks > 0:
                logger.warning(f"{failed_chunks}/{len(chunks)} chunks failed")
            
            return intersecting_indices
            
        except Exception as e:
            logger.error(f"Parallel spatial intersection failed: {e}")
            # Fallback to sequential processing
            logger.info("Falling back to sequential processing...")
            return self._sequential_spatial_intersection(target_gdf, source_gdf, use_bounds_check)
    
    def _sequential_spatial_intersection(
        self, 
        target_gdf: gpd.GeoDataFrame, 
        source_gdf: gpd.GeoDataFrame,
        use_bounds_check: bool = False
    ) -> List[int]:
        """Fallback sequential spatial intersection."""
        try:
            if use_bounds_check:
                source_bounds = source_gdf.total_bounds
                intersecting = []
                for idx, row in target_gdf.iterrows():
                    geom_bounds = row.geometry.bounds
                    if (geom_bounds[0] < source_bounds[2] and geom_bounds[2] > source_bounds[0] and
                        geom_bounds[1] < source_bounds[3] and geom_bounds[3] > source_bounds[1]):
                        intersecting.append(idx)
                return intersecting
            else:
                intersecting_gdf = gpd.sjoin(target_gdf, source_gdf, how='inner', predicate='intersects')
                return intersecting_gdf.index.unique().tolist()
        except Exception as e:
            logger.error(f"Sequential spatial intersection failed: {e}")
            return []


    def parallel_bootstrap_analysis(
        self, 
        group1_data: np.ndarray, 
        group2_data: np.ndarray, 
        n_iterations: int = 1000
    ) -> Dict[str, Any]:
        """
        Perform bootstrap analysis using parallel processing.
        
        Args:
            group1_data: Data for group 1
            group2_data: Data for group 2
            n_iterations: Number of bootstrap iterations
            
        Returns:
            Dictionary with bootstrap results
        """
        logger.info(f"Starting parallel bootstrap analysis: {n_iterations} iterations")
        
        # Determine optimal chunk size for iterations
        chunk_size = max(1, n_iterations // (self.max_workers * 4))
        
        # Prepare iteration chunks
        iteration_chunks = []
        options = {}
        
        remaining_iterations = n_iterations
        while remaining_iterations > 0:
            current_chunk_size = min(chunk_size, remaining_iterations)
            for _ in range(current_chunk_size):
                iteration_chunks.append((group1_data, group2_data, options))
            remaining_iterations -= current_chunk_size
        
        start_time = time.time()
        
        try:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(_bootstrap_worker, iteration_chunks))
            
            # Collect bootstrap statistics
            bootstrap_stats = []
            failed_iterations = 0
            
            for result in results:
                if result['success']:
                    bootstrap_stats.append(result['statistic'])
                else:
                    failed_iterations += 1
            
            duration = time.time() - start_time
            logger.info(f"Bootstrap analysis completed in {duration:.2f}s")
            
            if failed_iterations > 0:
                logger.warning(f"{failed_iterations}/{len(iteration_chunks)} iterations failed")
            
            return {
                'bootstrap_statistics': np.array(bootstrap_stats),
                'n_successful': len(bootstrap_stats),
                'n_failed': failed_iterations
            }
            
        except Exception as e:
            logger.error(f"Parallel bootstrap analysis failed: {e}")
            return {
                'bootstrap_statistics': np.array([]),
                'n_successful': 0,
                'n_failed': n_iterations
            }

    def parallel_permutation_test(
        self, 
        group1_data: np.ndarray, 
        group2_data: np.ndarray, 
        n_permutations: int = 1000
    ) -> Dict[str, Any]:
        """
        Perform permutation test using parallel processing.
        
        Args:
            group1_data: Data for group 1
            group2_data: Data for group 2  
            n_permutations: Number of permutations
            
        Returns:
            Dictionary with permutation test results
        """
        logger.info(f"Starting parallel permutation test: {n_permutations} permutations")
        
        # Combine data for permutation
        combined_data = np.concatenate([group1_data, group2_data])
        n1, n2 = len(group1_data), len(group2_data)
        
        # Prepare permutation chunks
        permutation_chunks = [(combined_data, n1, n2) for _ in range(n_permutations)]
        
        start_time = time.time()
        
        try:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                permutation_stats = list(executor.map(_permutation_worker, permutation_chunks))
            
            # Filter out failed results (NaN values)
            valid_stats = [stat for stat in permutation_stats if not np.isnan(stat)]
            failed_permutations = len(permutation_stats) - len(valid_stats)
            
            duration = time.time() - start_time
            logger.info(f"Permutation test completed in {duration:.2f}s")
            
            if failed_permutations > 0:
                logger.warning(f"{failed_permutations}/{n_permutations} permutations failed")
            
            return {
                'permutation_statistics': np.array(valid_stats),
                'n_successful': len(valid_stats),
                'n_failed': failed_permutations
            }
            
        except Exception as e:
            logger.error(f"Parallel permutation test failed: {e}")
            return {
                'permutation_statistics': np.array([]),
                'n_successful': 0,
                'n_failed': n_permutations
            }


def optimize_memory_usage():
    """
    Set optimal memory settings for large dataset processing.
    """
    # Set environment variables for better memory management
    os.environ['OMP_NUM_THREADS'] = str(min(8, multiprocessing.cpu_count()))  # Limit OpenMP threads
    os.environ['NUMEXPR_MAX_THREADS'] = str(min(8, multiprocessing.cpu_count()))  # Limit NumExpr
    
    # Configure pandas for better memory usage
    pd.set_option('mode.copy_on_write', True)  # Reduce memory copying
    
    logger.info(f"Optimized memory settings: OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}")


# Auto-optimize on import
optimize_memory_usage() 