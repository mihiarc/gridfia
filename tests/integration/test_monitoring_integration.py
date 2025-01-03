"""
Integration tests for the monitoring layer.
"""
import pytest
import time
from pathlib import Path
import geopandas as gpd
import pandas as pd
import json
import logging

from processing.pipeline_monitor import PipelineMonitor
from processing.parcel_processor import ParcelProcessor

def test_monitor_pipeline_integration(
    test_validator,
    test_processor,
    test_monitor,
    test_checkpoint_dir,
    test_metrics_dir,
    large_geodataframe
):
    """Test monitoring integration with processing pipeline."""
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
    
    # Process data
    result = processor.process_geodataframe(large_geodataframe)
    
    # Check monitoring data
    metrics = test_monitor.get_processing_metrics("test_dataset")
    assert metrics.processed_records == len(large_geodataframe)
    assert metrics.peak_memory_mb > 0
    assert metrics.avg_processing_time > 0
    
    # Check resource metrics
    resource_metrics = test_monitor.get_latest_resource_metrics()
    assert resource_metrics.cpu_percent >= 0
    assert resource_metrics.memory_percent >= 0
    assert resource_metrics.disk_percent >= 0
    
    # Stop monitoring
    test_monitor.stop_monitoring()

def test_metrics_collection_flow(
    test_validator,
    test_processor,
    test_monitor,
    test_checkpoint_dir,
    test_metrics_dir,
    test_data_dir,
    large_geodataframe
):
    """Test metrics collection and storage."""
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
    
    # Define processing stages
    def stage1(gdf):
        """First processing stage."""
        time.sleep(0.1)
        return gdf
    
    def stage2(gdf):
        """Second processing stage."""
        time.sleep(0.2)
        return gdf
    
    # Process data in stages
    result1 = processor.process_geodataframe(
        large_geodataframe,
        chunk_processor=stage1
    )
    
    result2 = processor.process_geodataframe(
        result1.processed_data,
        chunk_processor=stage2
    )
    
    # Save metrics
    metrics_path = test_data_dir / "pipeline_metrics.json"
    test_monitor.save_metrics(metrics_path)
    
    # Check saved metrics
    with open(metrics_path) as f:
        saved_metrics = json.load(f)
        assert "resource_metrics" in saved_metrics
        assert "processing_metrics" in saved_metrics
        assert len(saved_metrics["resource_metrics"]) > 0
    
    # Stop monitoring
    test_monitor.stop_monitoring()

def test_alert_handling_integration(
    test_validator,
    test_processor,
    test_monitor,
    test_checkpoint_dir,
    test_metrics_dir,
    large_geodataframe,
    caplog
):
    """Test alert generation and handling."""
    # Set up logging
    caplog.set_level(logging.WARNING)
    
    # Initialize pipeline with low thresholds
    test_monitor.alert_cpu_threshold = 10.0
    test_monitor.alert_memory_threshold = 10.0
    test_monitor.alert_disk_threshold = 10.0
    
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
    
    # Define resource-intensive function
    def intensive_process(gdf):
        """Resource intensive processing."""
        # Create memory pressure
        temp = []
        for _ in range(1000):
            temp.append(pd.DataFrame({"data": range(1000)}))
        time.sleep(0.1)
        return gdf
    
    # Process data
    result = processor.process_geodataframe(
        large_geodataframe,
        chunk_processor=intensive_process
    )
    
    # Check alerts
    assert any("High CPU usage" in record.message for record in caplog.records)
    assert any("High memory usage" in record.message for record in caplog.records)
    
    # Stop monitoring
    test_monitor.stop_monitoring()

def test_monitoring_lifecycle_integration(
    test_validator,
    test_processor,
    test_monitor,
    test_checkpoint_dir,
    test_metrics_dir,
    test_data_dir,
    large_geodataframe
):
    """Test complete monitoring lifecycle."""
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
    
    # Record initial state
    initial_metrics = test_monitor.get_latest_resource_metrics()
    
    # Process data
    result = processor.process_geodataframe(large_geodataframe)
    
    # Record processing state
    processing_metrics = test_monitor.get_latest_resource_metrics()
    
    # Generate report
    report_path = test_data_dir / "lifecycle_report.json"
    report = test_monitor.generate_report(report_path)
    
    # Stop monitoring
    test_monitor.stop_monitoring()
    
    # Record final state
    final_metrics = test_monitor.get_latest_resource_metrics()
    
    # Check lifecycle metrics
    assert initial_metrics.timestamp < processing_metrics.timestamp
    assert processing_metrics.timestamp < final_metrics.timestamp
    assert report["summary"]["total_duration"] > 0
    assert report["summary"]["total_records"] == len(large_geodataframe)

def test_multi_dataset_monitoring(
    test_validator,
    test_processor,
    test_monitor,
    test_checkpoint_dir,
    test_metrics_dir,
    large_geodataframe
):
    """Test monitoring of multiple datasets."""
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
    
    # Process multiple datasets
    datasets = {
        "dataset1": large_geodataframe.iloc[:500],
        "dataset2": large_geodataframe.iloc[500:]
    }
    
    for name, data in datasets.items():
        result = processor.process_geodataframe(
            data,
            dataset_name=name
        )
        
        # Check dataset metrics
        metrics = test_monitor.get_processing_metrics(name)
        assert metrics.processed_records == len(data)
        assert metrics.peak_memory_mb > 0
    
    # Generate summary
    summary = test_monitor.generate_summary()
    
    # Check summary
    assert len(summary["datasets"]) == 2
    assert all(
        dataset in summary["datasets"] 
        for dataset in ["dataset1", "dataset2"]
    )
    assert summary["total_records"] == len(large_geodataframe)
    
    # Stop monitoring
    test_monitor.stop_monitoring() 