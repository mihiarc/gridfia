"""
Integration tests for the error recovery layer.
"""
import pytest
import time
from pathlib import Path
import geopandas as gpd
import pandas as pd
import json

from processing.transaction_manager import TransactionManager
from processing.parcel_processor import ParcelProcessor
from processing.pipeline_monitor import PipelineMonitor

def test_transaction_pipeline_integration(
    test_transaction_manager,
    test_validator,
    test_processor,
    test_monitor,
    test_checkpoint_dir,
    test_metrics_dir,
    large_geodataframe,
    mock_database
):
    """Test transaction management in processing pipeline."""
    # Initialize pipeline
    processor = ParcelProcessor(
        db_connection="sqlite:///:memory:",
        validator=test_validator,
        transaction_manager=test_transaction_manager,
        chunked_processor=test_processor,
        monitor=test_monitor,
        checkpoint_dir=test_checkpoint_dir,
        metrics_dir=test_metrics_dir
    )
    
    # Start monitoring
    test_monitor.start_monitoring()
    
    # Process data with transaction
    with test_transaction_manager.transaction("test_table") as engine:
        result = processor.process_geodataframe(
            large_geodataframe,
            db_engine=engine
        )
        
        # Check transaction status
        active_transactions = test_transaction_manager.list_active_transactions()
        assert len(active_transactions) == 1
        assert active_transactions[0]["table_name"] == "test_table"
    
    # Verify transaction completed
    assert len(test_transaction_manager.list_active_transactions()) == 0
    
    # Check monitoring metrics
    metrics = test_monitor.get_processing_metrics("test_dataset")
    assert metrics.processed_records == len(large_geodataframe)
    
    # Stop monitoring
    test_monitor.stop_monitoring()

def test_recovery_scenario_handling(
    test_transaction_manager,
    test_validator,
    test_processor,
    test_monitor,
    test_checkpoint_dir,
    test_metrics_dir,
    large_geodataframe,
    mock_database
):
    """Test error recovery scenarios."""
    # Initialize pipeline
    processor = ParcelProcessor(
        db_connection="sqlite:///:memory:",
        validator=test_validator,
        transaction_manager=test_transaction_manager,
        chunked_processor=test_processor,
        monitor=test_monitor,
        checkpoint_dir=test_checkpoint_dir,
        metrics_dir=test_metrics_dir
    )
    
    # Start monitoring
    test_monitor.start_monitoring()
    
    # Define failing function
    def failing_process(gdf):
        """Function that fails after some processing."""
        if len(gdf) > test_processor.chunk_size / 2:
            raise ValueError("Simulated failure")
        return gdf
    
    # Process data with expected failure
    try:
        with test_transaction_manager.transaction("test_table") as engine:
            result = processor.process_geodataframe(
                large_geodataframe,
                chunk_processor=failing_process,
                db_engine=engine
            )
    except ValueError:
        pass
    
    # Check recovery state
    assert len(test_transaction_manager.list_active_transactions()) == 0
    
    # Verify checkpoint creation
    checkpoint_files = list(
        test_transaction_manager.checkpoint_dir.glob("*.json")
    )
    assert len(checkpoint_files) > 0
    
    # Check monitoring metrics
    metrics = test_monitor.get_processing_metrics("test_dataset")
    assert metrics.processing_errors > 0
    
    # Stop monitoring
    test_monitor.stop_monitoring()

def test_checkpoint_recovery_flow(
    test_transaction_manager,
    test_validator,
    test_processor,
    test_monitor,
    test_checkpoint_dir,
    test_metrics_dir,
    large_geodataframe,
    mock_database,
    test_data_dir
):
    """Test recovery from checkpoints."""
    # Initialize pipeline
    processor = ParcelProcessor(
        db_connection="sqlite:///:memory:",
        validator=test_validator,
        transaction_manager=test_transaction_manager,
        chunked_processor=test_processor,
        monitor=test_monitor,
        checkpoint_dir=test_checkpoint_dir,
        metrics_dir=test_metrics_dir
    )
    
    # Start monitoring
    test_monitor.start_monitoring()
    
    # Create initial state
    with test_transaction_manager.transaction("test_table") as engine:
        # Create table
        with engine.connect() as conn:
            conn.execute("""
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY,
                value TEXT
            );
            """)
            conn.commit()
    
    # Process with checkpoint creation
    checkpoint_path = test_data_dir / "checkpoints/test_checkpoint.json"
    try:
        with test_transaction_manager.transaction("test_table") as engine:
            result = processor.process_geodataframe(
                large_geodataframe,
                db_engine=engine,
                checkpoint_path=checkpoint_path
            )
            raise Exception("Simulated failure")
    except:
        pass
    
    # Verify checkpoint
    assert checkpoint_path.exists()
    with open(checkpoint_path) as f:
        checkpoint = json.load(f)
        assert checkpoint["table_name"] == "test_table"
        assert "state" in checkpoint
        assert "error" in checkpoint["state"]
    
    # Check monitoring metrics
    metrics = test_monitor.get_processing_metrics("test_dataset")
    assert metrics.processing_errors > 0
    
    # Stop monitoring
    test_monitor.stop_monitoring()

def test_transaction_monitoring_integration(
    test_transaction_manager,
    test_validator,
    test_processor,
    test_monitor,
    test_checkpoint_dir,
    test_metrics_dir,
    large_geodataframe,
    mock_database
):
    """Test transaction monitoring integration."""
    # Initialize pipeline
    processor = ParcelProcessor(
        db_connection="sqlite:///:memory:",
        validator=test_validator,
        transaction_manager=test_transaction_manager,
        chunked_processor=test_processor,
        monitor=test_monitor,
        checkpoint_dir=test_checkpoint_dir,
        metrics_dir=test_metrics_dir
    )
    
    # Start monitoring
    test_monitor.start_monitoring()
    
    # Process with transaction monitoring
    with test_transaction_manager.transaction("test_table") as engine:
        # Get initial transaction state
        initial_state = test_transaction_manager.get_transaction_status(
            list(test_transaction_manager.active_transactions.keys())[0]
        )
        
        # Process data
        result = processor.process_geodataframe(
            large_geodataframe,
            db_engine=engine
        )
        
        # Get final transaction state
        final_state = test_transaction_manager.get_transaction_status(
            list(test_transaction_manager.active_transactions.keys())[0]
        )
    
    # Check transaction monitoring
    assert initial_state["start_time"] < final_state["start_time"]
    assert "processed_records" in final_state
    
    # Verify monitoring metrics
    metrics = test_monitor.get_processing_metrics("test_dataset")
    assert metrics.processed_records == len(large_geodataframe)
    
    # Stop monitoring
    test_monitor.stop_monitoring() 