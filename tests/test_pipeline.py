"""Tests for the data pipeline functionality."""
import os
from pathlib import Path
import pytest
from unittest.mock import Mock, patch

from scripts.run_pipeline import (
    get_db_connection,
    process_parcel_data,
    process_heirs_property,
    main
)

@pytest.fixture
def mock_env_vars():
    """Set up mock environment variables."""
    env_vars = {
        'POSTGRES_USER': 'test_user',
        'POSTGRES_PASSWORD': 'test_pass',
        'POSTGRES_HOST': 'localhost',
        'POSTGRES_PORT': '5432',
        'POSTGRES_DB': 'test_db'
    }
    with patch.dict(os.environ, env_vars):
        yield env_vars

@pytest.fixture
def mock_processor():
    """Create a mock ParcelProcessor."""
    processor = Mock()
    processor.process_parquet.return_value = True
    processor.process_gdb.return_value = True
    return processor

def test_get_db_connection(mock_env_vars):
    """Test database connection string generation."""
    conn_str = get_db_connection()
    assert conn_str == (
        "postgresql://test_user:test_pass@localhost:5432/test_db"
    )

def test_get_db_connection_missing_vars():
    """Test handling of missing environment variables."""
    with pytest.raises(ValueError) as exc:
        get_db_connection()
    assert "Missing required environment variables" in str(exc.value)

@patch('pathlib.Path.exists')
def test_process_parcel_data_success(mock_exists, mock_processor):
    """Test successful parcel data processing."""
    mock_exists.return_value = True
    data_dir = Path("/test/data")
    
    success = process_parcel_data(mock_processor, data_dir)
    
    assert success is True
    assert mock_processor.process_parquet.call_count == 2
    assert mock_processor.process_gdb.call_count == 1

@patch('pathlib.Path.exists')
def test_process_parcel_data_missing_files(mock_exists, mock_processor):
    """Test parcel processing with missing files."""
    mock_exists.return_value = False
    data_dir = Path("/test/data")
    
    success = process_parcel_data(mock_processor, data_dir)
    
    assert success is False
    assert mock_processor.process_parquet.call_count == 0
    assert mock_processor.process_gdb.call_count == 0

@patch('pathlib.Path.exists')
def test_process_heirs_property_success(mock_exists, mock_processor):
    """Test successful heirs property data processing."""
    mock_exists.return_value = True
    data_dir = Path("/test/data")
    
    success = process_heirs_property(mock_processor, data_dir)
    
    assert success is True
    assert mock_processor.process_parquet.call_count == 1
    assert mock_processor.process_gdb.call_count == 1

@patch('pathlib.Path.exists')
def test_process_heirs_property_missing_files(mock_exists, mock_processor):
    """Test heirs property processing with missing files."""
    mock_exists.return_value = False
    data_dir = Path("/test/data")
    
    success = process_heirs_property(mock_processor, data_dir)
    
    assert success is False
    assert mock_processor.process_parquet.call_count == 0
    assert mock_processor.process_gdb.call_count == 0

@patch('scripts.run_pipeline.get_db_connection')
@patch('scripts.run_pipeline.ParcelProcessor')
@patch('pathlib.Path.exists')
def test_main_success(mock_exists, mock_processor_class, mock_get_conn):
    """Test successful pipeline execution."""
    mock_exists.return_value = True
    mock_processor = Mock()
    mock_processor.process_parquet.return_value = True
    mock_processor.process_gdb.return_value = True
    mock_processor_class.return_value = mock_processor
    mock_get_conn.return_value = "test_conn"
    
    with pytest.raises(SystemExit) as exc:
        main()
    
    assert exc.value.code == 0
    assert mock_processor.process_parquet.call_count > 0
    assert mock_processor.process_gdb.call_count > 0

@patch('scripts.run_pipeline.get_db_connection')
@patch('scripts.run_pipeline.ParcelProcessor')
@patch('pathlib.Path.exists')
def test_main_failure(mock_exists, mock_processor_class, mock_get_conn):
    """Test pipeline execution with failures."""
    mock_exists.return_value = True
    mock_processor = Mock()
    mock_processor.process_parquet.return_value = False
    mock_processor.process_gdb.return_value = False
    mock_processor_class.return_value = mock_processor
    mock_get_conn.return_value = "test_conn"
    
    with pytest.raises(SystemExit) as exc:
        main()
    
    assert exc.value.code == 1 