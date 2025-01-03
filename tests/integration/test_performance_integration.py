"""
Integration tests for the performance optimization layer.
"""
import pytest
import time
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np

from processing.chunked_processor import ChunkedProcessor
from processing.pipeline_monitor import PipelineMonitor
from processing.parcel_processor import ParcelProcessor

def test_chunked_processing_pipeline(
    test_validator,
    test_processor,
    test_monitor,
    test_checkpoint_dir,
    test_metrics_dir,
    large_geodataframe
):
    """Test chunked processing integration with monitoring."""
    # Initialize pipeline
    processor = ParcelProcessor(
        db_connection="sqlite:///:memory:",
        validator=test_validator,
        chunked_processor=test_processor,
        monitor=test_monitor,
        checkpoint_dir=test_checkpoint_dir,
        metrics_dir=test_metrics_dir
    )
    
    # Start monitoring
    test_monitor.start_monitoring()
    
    # Define processing function
    def process_chunk(gdf):
        """Simple processing function."""
        time.sleep(0.1)  # Simulate work
        return gdf
    
    # Process data in chunks
    result = processor.process_geodataframe(
        large_geodataframe,
        chunk_processor=process_chunk
    )
    
    # Check processing results
    assert len(result.processed_data) == len(large_geodataframe)
    
    # Check monitoring metrics
    metrics = test_monitor.get_processing_metrics("test_dataset")
    assert metrics.processed_records == len(large_geodataframe)
    assert metrics.peak_memory_mb > 0
    
    # Stop monitoring
    test_monitor.stop_monitoring()

def test_memory_management_integration(
    test_validator,
    test_processor,
    test_monitor,
    test_checkpoint_dir,
    test_metrics_dir,
    large_geodataframe
):
    """Test memory management across pipeline."""
    # Create processor with low memory limit
    processor = ParcelProcessor(
        db_connection="sqlite:///:memory:",
        validator=test_validator,
        chunked_processor=ChunkedProcessor(
            chunk_size=100,
            max_workers=2,
            memory_limit_mb=50  # Very low limit
        ),
        monitor=test_monitor,
        checkpoint_dir=test_checkpoint_dir,
        metrics_dir=test_metrics_dir
    )
    
    # Start monitoring
    test_monitor.start_monitoring()
    
    # Define memory-intensive function
    def memory_intensive(gdf):
        """Function that uses significant memory."""
        # Create temporary large data
        temp = pd.concat([gdf] * 5)
        time.sleep(0.1)
        return gdf
    
    # Process data
    result = processor.process_geodataframe(
        large_geodataframe,
        chunk_processor=memory_intensive
    )
    
    # Check memory metrics
    metrics = test_monitor.get_processing_metrics("test_dataset")
    assert metrics.peak_memory_mb > 0
    assert any(
        usage > processor.chunked_processor.memory_limit_mb 
        for usage in metrics.memory_usage.values()
    )
    
    # Stop monitoring
    test_monitor.stop_monitoring()

def test_parallel_processing_integration(
    test_validator,
    test_processor,
    test_monitor,
    test_checkpoint_dir,
    test_metrics_dir,
    large_geodataframe
):
    """Test parallel processing with monitoring."""
    # Create processor with multiple workers
    processor = ParcelProcessor(
        db_connection="sqlite:///:memory:",
        validator=test_validator,
        chunked_processor=ChunkedProcessor(
            chunk_size=100,
            max_workers=4,
            memory_limit_mb=500
        ),
        monitor=test_monitor,
        checkpoint_dir=test_checkpoint_dir,
        metrics_dir=test_metrics_dir
    )
    
    # Start monitoring
    test_monitor.start_monitoring()
    
    # Define CPU-intensive function
    def cpu_intensive(gdf):
        """Function that uses CPU."""
        # Perform calculations
        gdf['calc'] = np.random.random(len(gdf))
        gdf['calc2'] = gdf['calc'].rolling(10).mean()
        time.sleep(0.1)
        return gdf
    
    # Process data
    start_time = time.time()
    result = processor.process_geodataframe(
        large_geodataframe,
        chunk_processor=cpu_intensive
    )
    processing_time = time.time() - start_time
    
    # Check processing metrics
    metrics = test_monitor.get_processing_metrics("test_dataset")
    assert metrics.processed_records == len(large_geodataframe)
    assert metrics.avg_processing_time > 0
    
    # Check CPU metrics
    resource_metrics = test_monitor.get_latest_resource_metrics()
    assert resource_metrics.cpu_percent > 0
    
    # Stop monitoring
    test_monitor.stop_monitoring()

def test_resource_optimization_integration(
    test_validator,
    test_processor,
    test_monitor,
    test_checkpoint_dir,
    test_metrics_dir,
    test_data_dir,
    large_geodataframe
):
    """Test resource optimization with reporting."""
    # Initialize processor
    processor = ParcelProcessor(
        db_connection="sqlite:///:memory:",
        validator=test_validator,
        chunked_processor=test_processor,
        monitor=test_monitor,
        checkpoint_dir=test_checkpoint_dir,
        metrics_dir=test_metrics_dir
    )
    
    # Set up monitoring with low thresholds
    test_monitor.alert_cpu_threshold = 50.0
    test_monitor.alert_memory_threshold = 50.0
    
    # Start monitoring
    test_monitor.start_monitoring()
    
    # Define resource-intensive function
    def resource_intensive(gdf):
        """Function that uses both CPU and memory."""
        # Create temporary data
        temp = []
        for _ in range(1000):
            temp.append(np.random.random(1000))
        gdf['calc'] = np.random.random(len(gdf))
        time.sleep(0.1)
        return gdf
    
    # Process data
    result = processor.process_geodataframe(
        large_geodataframe,
        chunk_processor=resource_intensive
    )
    
    # Generate resource report
    report_path = test_data_dir / "resource_report.json"
    report = test_monitor.generate_report(report_path)
    
    # Check report contents
    assert "resource_metrics" in report
    assert "averages" in report["resource_metrics"]
    assert "peaks" in report["resource_metrics"]
    assert report["resource_metrics"]["peaks"]["cpu_percent"] > 0
    assert report["resource_metrics"]["peaks"]["memory_percent"] > 0
    
    # Stop monitoring
    test_monitor.stop_monitoring() 