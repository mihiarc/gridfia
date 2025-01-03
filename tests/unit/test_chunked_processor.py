"""
Unit tests for the ChunkedProcessor class.
"""
import pytest
import time
import multiprocessing
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from shapely.geometry import Polygon

from processing.chunked_processor import ChunkedProcessor, ProcessingStats

# Configure multiprocessing to use 'spawn'
if hasattr(multiprocessing, 'set_start_method'):
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # Method already set
        pass

def simple_process_func(gdf):
    """Simple processing function."""
    time.sleep(0.1)  # Simulate work
    return gdf

def memory_intensive_func(gdf):
    """Function that uses memory."""
    # Create some temporary data to use memory
    temp = pd.concat([gdf] * 10)
    time.sleep(0.1)  # Give time for memory monitoring
    return gdf

def slow_process_func(gdf):
    """Slow processing function."""
    time.sleep(0.1)
    return gdf

def cpu_intensive_func(gdf):
    """CPU intensive function."""
    # Perform some calculations
    gdf['calc'] = np.random.random(len(gdf))
    time.sleep(0.1)
    return gdf

def memory_limit_func(gdf):
    """Memory intensive function for testing limits."""
    # Create large temporary data
    temp = pd.concat([gdf] * 100)
    time.sleep(0.1)
    return gdf

def collect_chunks_func(gdf):
    """Collect processed chunks."""
    time.sleep(0.1)
    return gdf

def failing_func(gdf):
    """Function that fails on certain chunks."""
    if len(gdf) > 50:  # Fail on larger chunks
        raise ValueError("Simulated failure")
    return gdf

def error_func(gdf):
    """Function that raises an error."""
    raise ValueError("Test error")

def test_chunk_size_calculation(test_processor, large_geodataframe):
    """Test chunk size determination."""
    # Test with default chunk size
    chunks = [
        large_geodataframe.iloc[i:i + test_processor.chunk_size]
        for i in range(0, len(large_geodataframe), test_processor.chunk_size)
    ]
    
    assert len(chunks) == len(large_geodataframe) // test_processor.chunk_size + (
        1 if len(large_geodataframe) % test_processor.chunk_size > 0 else 0
    )
    assert all(len(chunk) <= test_processor.chunk_size for chunk in chunks)

def test_parallel_processing(test_processor, large_geodataframe):
    """Test parallel processing functionality."""
    # Process with multiple workers
    result, stats = test_processor.process_geodataframe(
        large_geodataframe,
        simple_process_func
    )
    
    assert result is not None
    assert len(result) == len(large_geodataframe)
    assert stats.processed_chunks > 0
    assert stats.processed_records == len(large_geodataframe)

def test_memory_monitoring(test_processor, large_geodataframe):
    """Test memory usage monitoring."""
    result, stats = test_processor.process_geodataframe(
        large_geodataframe,
        memory_intensive_func
    )
    
    assert result is not None
    assert len(stats.memory_usage) > 0
    assert all(
        isinstance(usage, float) 
        for usage in stats.memory_usage.values()
    )
    assert max(stats.memory_usage.values()) > 0

def test_progress_tracking(test_processor, large_geodataframe):
    """Test progress reporting."""
    result, stats = test_processor.process_geodataframe(
        large_geodataframe,
        slow_process_func,
        progress_bar=True
    )
    
    assert result is not None
    assert len(result) == len(large_geodataframe)
    assert stats.processed_chunks > 0
    assert stats.processed_records == len(large_geodataframe)

def test_processing_stats(test_processor, large_geodataframe):
    """Test processing statistics generation."""
    result, stats = test_processor.process_geodataframe(
        large_geodataframe,
        failing_func
    )
    
    assert isinstance(stats, ProcessingStats)
    assert stats.start_time is not None
    assert stats.end_time is not None
    assert stats.total_chunks > 0
    assert len(stats.failed_chunks) > 0
    assert stats.processed_records < len(large_geodataframe)

def test_error_handling(test_processor, large_geodataframe):
    """Test error handling in processing."""
    result, stats = test_processor.process_geodataframe(
        large_geodataframe,
        error_func
    )
    
    assert result is None
    assert len(stats.failed_chunks) == stats.total_chunks
    assert stats.processed_records == 0

def test_process_parquet(test_processor, test_data_dir):
    """Test processing of parquet files."""
    # Create a simple test GeoDataFrame
    data = {
        'id': range(10),
        'value': range(10),
        'geometry': [
            Polygon([
                (0, 0), (0, 1), (1, 1), (1, 0), (0, 0)
            ])
        ] * 10
    }
    test_gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    
    # Convert to DataFrame with WKT geometry
    df = pd.DataFrame(test_gdf.drop(columns=['geometry']))
    df['geometry'] = test_gdf.geometry.apply(lambda x: x.wkt)
    
    # Save as parquet
    parquet_path = test_data_dir / "test.parquet"
    df.to_parquet(parquet_path, index=False)
    
    # Process parquet file
    result, stats = test_processor.process_parquet(
        str(parquet_path),
        simple_process_func
    )
    
    assert result is not None
    assert len(result) == len(test_gdf)
    assert stats.processed_chunks > 0
    assert stats.processed_records == len(test_gdf)

def test_worker_management(test_processor, large_geodataframe):
    """Test worker pool management."""
    # Test with different worker counts
    for workers in [1, 2, 4]:
        processor = ChunkedProcessor(
            chunk_size=100,
            max_workers=workers,
            memory_limit_mb=500
        )

        result, stats = processor.process_geodataframe(
            large_geodataframe,
            cpu_intensive_func
        )

        assert result is not None
        assert len(result) == len(large_geodataframe)
        assert stats.processed_chunks > 0
        assert stats.processed_records == len(large_geodataframe)

def test_memory_limit_handling(large_geodataframe):
    """Test handling of memory limits."""
    # Create processor with low memory limit
    processor = ChunkedProcessor(
        chunk_size=100,
        max_workers=2,
        memory_limit_mb=10  # Very low limit
    )

    result, stats = processor.process_geodataframe(
        large_geodataframe,
        memory_limit_func
    )

    assert result is not None
    assert len(result) == len(large_geodataframe)
    assert stats.processed_chunks > 0
    assert stats.processed_records == len(large_geodataframe)

def test_concurrent_processing(test_processor, large_geodataframe):
    """Test concurrent processing capabilities."""
    result, stats = test_processor.process_geodataframe(
        large_geodataframe,
        collect_chunks_func
    )

    assert result is not None
    assert len(result) == len(large_geodataframe)
    assert stats.processed_chunks > 0
    assert stats.processed_records == len(large_geodataframe)

def test_stats_serialization(test_processor, large_geodataframe):
    """Test serialization of processing stats."""
    result, stats = test_processor.process_geodataframe(
        large_geodataframe,
        simple_process_func
    )
    
    # Convert stats to dict
    stats_dict = stats.to_dict()
    
    # Check dict structure
    assert "start_time" in stats_dict
    assert "end_time" in stats_dict
    assert "total_chunks" in stats_dict
    assert "processed_chunks" in stats_dict
    assert "total_records" in stats_dict
    assert "processed_records" in stats_dict
    assert "failed_chunks" in stats_dict
    assert "memory_usage" in stats_dict
    assert "success_rate" in stats_dict 