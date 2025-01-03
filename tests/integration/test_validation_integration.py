"""
Integration tests for the validation layer.
"""
import pytest
from pathlib import Path
import geopandas as gpd
import pandas as pd
import json

from processing.data_validator import DataValidator
from processing.parcel_processor import ParcelProcessor
from processing.pipeline_monitor import PipelineMonitor

def test_validation_pipeline_integration(
    test_validator,
    test_processor,
    test_monitor,
    test_checkpoint_dir,
    test_metrics_dir,
    large_geodataframe
):
    """Test validation integration with processing pipeline."""
    # Initialize pipeline components
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
    
    # Process data with validation
    result = processor.process_geodataframe(large_geodataframe)
    
    # Check validation results
    assert result.validation_success
    assert len(result.validation_report["issues"]) == 0
    
    # Check monitoring metrics
    metrics = test_monitor.get_processing_metrics("test_dataset")
    assert metrics.validation_errors == 0
    assert metrics.processed_records == len(large_geodataframe)
    
    # Stop monitoring
    test_monitor.stop_monitoring()

def test_validation_error_handling(
    test_validator,
    test_processor,
    test_monitor,
    test_checkpoint_dir,
    test_metrics_dir,
    large_geodataframe
):
    """Test handling of validation errors in pipeline."""
    # Create invalid data
    invalid_data = large_geodataframe.copy()
    invalid_data.drop(columns=["area"], inplace=True)
    
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
    
    # Process invalid data
    result = processor.process_geodataframe(invalid_data)
    
    # Check validation failure
    assert not result.validation_success
    assert len(result.validation_report["issues"]) > 0
    assert any(
        issue["type"] == "missing_fields" 
        for issue in result.validation_report["issues"]
    )
    
    # Check monitoring metrics
    metrics = test_monitor.get_processing_metrics("test_dataset")
    assert metrics.validation_errors > 0
    assert metrics.processed_records == 0  # Should not process invalid data
    
    # Stop monitoring
    test_monitor.stop_monitoring()

def test_validation_reporting_flow(
    test_validator,
    test_processor,
    test_monitor,
    test_checkpoint_dir,
    test_metrics_dir,
    test_data_dir,
    large_geodataframe
):
    """Test validation reporting through pipeline."""
    # Initialize pipeline
    processor = ParcelProcessor(
        db_connection="sqlite:///:memory:",
        validator=test_validator,
        chunked_processor=test_processor,
        monitor=test_monitor,
        checkpoint_dir=test_checkpoint_dir,
        metrics_dir=test_metrics_dir
    )
    
    # Set up report paths
    validation_report_path = test_data_dir / "validation_report.json"
    processing_report_path = test_data_dir / "processing_report.json"
    
    # Process data
    result = processor.process_geodataframe(
        large_geodataframe,
        validation_report_path=validation_report_path,
        processing_report_path=processing_report_path
    )
    
    # Check report files
    assert validation_report_path.exists()
    assert processing_report_path.exists()
    
    # Verify report contents
    with open(validation_report_path) as f:
        validation_report = json.load(f)
        assert "dataset" in validation_report
        assert "timestamp" in validation_report
        assert "record_count" in validation_report
        assert "issues" in validation_report
    
    with open(processing_report_path) as f:
        processing_report = json.load(f)
        assert "validation_metrics" in processing_report
        assert "processing_metrics" in processing_report
        assert "resource_metrics" in processing_report 