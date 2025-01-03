"""
Unit tests for the PipelineMonitor class.
"""
import pytest
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import psutil

from processing.pipeline_monitor import PipelineMonitor, ResourceMetrics, ProcessingMetrics

def test_resource_monitoring(test_monitor):
    """Test resource metrics collection."""
    # Start monitoring
    test_monitor.start_monitoring()
    time.sleep(2)  # Wait for some metrics to be collected
    test_monitor.stop_monitoring()
    
    # Check metrics
    assert len(test_monitor.resource_metrics) > 0
    for metric in test_monitor.resource_metrics:
        assert isinstance(metric, ResourceMetrics)
        assert metric.cpu_percent >= 0
        assert metric.memory_percent >= 0
        assert metric.memory_used_mb > 0
        assert metric.disk_percent >= 0
        assert metric.disk_used_gb > 0

def test_alert_thresholds(test_monitor, caplog):
    """Test alert generation."""
    # Create monitor with low thresholds
    monitor = PipelineMonitor(
        metrics_dir=test_monitor.metrics_dir,
        resource_interval=1,
        alert_cpu_threshold=0.1,  # Very low threshold
        alert_memory_threshold=0.1,
        alert_disk_threshold=0.1
    )
    
    # Start monitoring
    monitor.start_monitoring()
    time.sleep(2)  # Wait for alerts to be generated
    monitor.stop_monitoring()
    
    # Check for alerts in logs
    assert any("High CPU usage" in record.message for record in caplog.records)
    assert any("High memory usage" in record.message for record in caplog.records)
    assert any("High disk usage" in record.message for record in caplog.records)

def test_metrics_storage(test_monitor, mock_resource_metrics):
    """Test metrics persistence."""
    # Add some metrics
    metric = ResourceMetrics(
        timestamp=datetime.now().isoformat(),
        **mock_resource_metrics
    )
    test_monitor.resource_metrics.append(metric)
    
    # Save metrics
    test_monitor.save_metrics()
    
    # Check saved files
    metric_files = list(test_monitor.metrics_dir.glob("resource_metrics_*.json"))
    assert len(metric_files) > 0
    
    # Verify content
    with open(metric_files[0]) as f:
        saved_metrics = json.load(f)
        assert len(saved_metrics) > 0
        assert all(
            key in saved_metrics[0] 
            for key in mock_resource_metrics.keys()
        )

def test_report_generation(test_monitor, mock_processing_metrics):
    """Test report creation."""
    # Add some processing metrics
    test_monitor.processing_metrics["test_dataset"] = ProcessingMetrics(
        **mock_processing_metrics
    )
    
    # Generate report
    report_path = test_monitor.metrics_dir / "report.json"
    report = test_monitor.generate_report(report_path)
    
    # Check report structure
    assert "timestamp" in report
    assert "resource_metrics" in report
    assert "processing_metrics" in report
    assert "summary" in report
    
    # Check report content
    assert report["processing_metrics"]["test_dataset"]["total_records"] == 1000
    assert report["processing_metrics"]["test_dataset"]["processed_records"] == 800

def test_monitor_lifecycle(test_monitor):
    """Test monitor start/stop."""
    # Check initial state
    assert not test_monitor._monitoring
    assert test_monitor._monitor_thread is None
    
    # Start monitoring
    test_monitor.start_monitoring()
    assert test_monitor._monitoring
    assert test_monitor._monitor_thread is not None
    assert test_monitor._monitor_thread.is_alive()
    
    # Stop monitoring
    test_monitor.stop_monitoring()
    assert not test_monitor._monitoring
    assert not test_monitor._monitor_thread.is_alive()

def test_processing_metrics_update(test_monitor):
    """Test processing metrics updates."""
    dataset_name = "test_dataset"
    
    # Initialize metrics
    metrics = test_monitor.start_processing_metrics(
        dataset_name=dataset_name,
        total_records=1000
    )
    
    # Update metrics
    test_monitor.update_processing_metrics(
        dataset_name=dataset_name,
        processed_records=100,
        failed_records=10,
        validation_errors=5,
        processing_errors=2,
        processing_time=0.5,
        memory_mb=1000.0
    )
    
    # Check updates
    updated_metrics = test_monitor.get_processing_metrics(dataset_name)
    assert updated_metrics.processed_records == 100
    assert updated_metrics.failed_records == 10
    assert updated_metrics.validation_errors == 5
    assert updated_metrics.processing_errors == 2
    assert updated_metrics.avg_processing_time == 0.5
    assert updated_metrics.peak_memory_mb == 1000.0

def test_metrics_queue(test_monitor):
    """Test real-time metrics queue."""
    # Start monitoring
    test_monitor.start_monitoring()
    time.sleep(2)  # Wait for metrics
    
    # Get latest metrics
    metrics = test_monitor.get_latest_resource_metrics()
    assert metrics is not None
    assert isinstance(metrics, ResourceMetrics)
    
    # Stop monitoring
    test_monitor.stop_monitoring()

def test_resource_calculation(test_monitor):
    """Test resource usage calculation."""
    # Start monitoring
    test_monitor.start_monitoring()
    
    # Create some CPU/memory load
    data = []
    for _ in range(1000000):
        data.append(np.random.random())
    time.sleep(1)
    
    # Get metrics
    metrics = test_monitor.get_latest_resource_metrics()
    assert metrics.cpu_percent > 0
    assert metrics.memory_used_mb > 0
    
    # Cleanup
    test_monitor.stop_monitoring()
    del data

def test_metrics_aggregation(test_monitor):
    """Test metrics aggregation in reports."""
    # Add multiple metrics
    for i in range(5):
        test_monitor.resource_metrics.append(ResourceMetrics(
            timestamp=datetime.now().isoformat(),
            cpu_percent=10.0 * i,
            memory_percent=20.0 * i,
            memory_used_mb=1000.0 * i,
            disk_percent=30.0 * i,
            disk_used_gb=100.0 * i
        ))
    
    # Generate report
    report = test_monitor.generate_report()
    
    # Check aggregations
    averages = report["resource_metrics"]["averages"]
    peaks = report["resource_metrics"]["peaks"]
    
    assert 0 <= averages["cpu_percent"] <= 100
    assert 0 <= averages["memory_percent"] <= 100
    assert peaks["cpu_percent"] >= averages["cpu_percent"]
    assert peaks["memory_percent"] >= averages["memory_percent"]

def test_error_handling(test_monitor, caplog):
    """Test error handling in monitoring."""
    def raise_error():
        raise Exception("Test error")
    
    # Patch _monitor_resources to raise error
    test_monitor._monitor_resources = raise_error
    
    # Start monitoring
    test_monitor.start_monitoring()
    time.sleep(1)
    test_monitor.stop_monitoring()
    
    # Check error was logged
    assert any("Error monitoring resources" in record.message for record in caplog.records)

def test_concurrent_updates(test_monitor):
    """Test concurrent metric updates."""
    import threading
    
    def update_metrics():
        for i in range(10):
            test_monitor.update_processing_metrics(
                dataset_name="test_dataset",
                processed_records=10,
                processing_time=0.1
            )
            time.sleep(0.1)
    
    # Start multiple update threads
    threads = [
        threading.Thread(target=update_metrics)
        for _ in range(3)
    ]
    
    # Initialize metrics
    test_monitor.start_processing_metrics(
        dataset_name="test_dataset",
        total_records=1000
    )
    
    # Run threads
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    
    # Check final metrics
    metrics = test_monitor.get_processing_metrics("test_dataset")
    assert metrics.processed_records == 300  # 10 updates * 10 records * 3 threads 