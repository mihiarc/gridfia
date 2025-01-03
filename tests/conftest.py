"""Test configuration and fixtures."""
import os
import tempfile
from pathlib import Path
import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
from sqlalchemy import create_engine, text

from processing.data_validator import DataValidator
from processing.chunked_processor import ChunkedProcessor
from processing.pipeline_monitor import PipelineMonitor
from processing.transaction_manager import TransactionManager

@pytest.fixture
def test_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def test_checkpoint_dir(test_data_dir):
    """Create a temporary directory for checkpoints."""
    checkpoint_dir = test_data_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir

@pytest.fixture
def test_metrics_dir(test_data_dir):
    """Create a temporary directory for metrics."""
    metrics_dir = test_data_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return metrics_dir

@pytest.fixture
def mock_database():
    """Create an in-memory SQLite database."""
    engine = create_engine("sqlite:///:memory:")
    
    # Create test tables
    with engine.connect() as conn:
        # Create test_table
        conn.execute(text("""
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY,
                value TEXT
            )
        """))
        
        # Create test_backup
        conn.execute(text("""
            CREATE TABLE test_backup (
                id INTEGER PRIMARY KEY,
                value TEXT
            )
        """))
        
        # Create test_checkpoint
        conn.execute(text("""
            CREATE TABLE test_checkpoint (
                id INTEGER PRIMARY KEY,
                value TEXT
            )
        """))
        
        # Create test_rollback
        conn.execute(text("""
            CREATE TABLE test_rollback (
                id INTEGER PRIMARY KEY,
                value TEXT
            )
        """))
        
        # Create test_status
        conn.execute(text("""
            CREATE TABLE test_status (
                id INTEGER PRIMARY KEY,
                value TEXT
            )
        """))
        
        # Create test_concurrent_1 and test_concurrent_2
        conn.execute(text("""
            CREATE TABLE test_concurrent_1 (
                id INTEGER PRIMARY KEY,
                value TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE test_concurrent_2 (
                id INTEGER PRIMARY KEY,
                value TEXT
            )
        """))
        
        # Create test_isolation
        conn.execute(text("""
            CREATE TABLE test_isolation (
                id INTEGER PRIMARY KEY,
                value TEXT
            )
        """))
        
        # Create test_cleanup
        conn.execute(text("""
            CREATE TABLE test_cleanup (
                id INTEGER PRIMARY KEY,
                value TEXT
            )
        """))
        
        # Create test_timing
        conn.execute(text("""
            CREATE TABLE test_timing (
                id INTEGER PRIMARY KEY,
                value TEXT
            )
        """))
        
        # Create test_error_format
        conn.execute(text("""
            CREATE TABLE test_error_format (
                id INTEGER PRIMARY KEY,
                value TEXT
            )
        """))
        
        conn.commit()
    
    return engine

@pytest.fixture
def test_validator():
    """Create a test validator."""
    return DataValidator(
        required_fields=["geometry", "area"],
        geometry_type="Polygon",
        srid=2264
    )

@pytest.fixture
def test_processor():
    """Create a test processor."""
    return ChunkedProcessor(
        chunk_size=100,
        max_workers=2,
        memory_limit_mb=500
    )

@pytest.fixture
def test_monitor(test_metrics_dir):
    """Create a test monitor."""
    return PipelineMonitor(
        metrics_dir=test_metrics_dir,
        resource_interval=1,
        alert_cpu_threshold=80.0,
        alert_memory_threshold=80.0,
        alert_disk_threshold=80.0
    )

@pytest.fixture
def test_transaction_manager(mock_database, test_checkpoint_dir):
    """Create a test transaction manager."""
    return TransactionManager(
        db_connection="sqlite:///:memory:",
        checkpoint_dir=test_checkpoint_dir
    )

@pytest.fixture
def large_geodataframe():
    """Create a large test GeoDataFrame."""
    n_rows = 1000
    
    # Create DataFrame first
    df = pd.DataFrame({
        "id": range(n_rows),
        "parcel_id": [f"P{i:03d}" for i in range(n_rows)],
        "area": [i * 100.0 for i in range(n_rows)],
        "county": ["County1"] * n_rows,
    })
    
    # Create geometry column separately
    geometry = [
        Polygon([
            (0, 0), (0, 1), (1, 1), (1, 0), (0, 0)
        ])
    ] * n_rows
    
    # Create GeoDataFrame using public API
    return gpd.GeoDataFrame(
        df,
        geometry=geometry,
        crs="EPSG:2264"
    ) 