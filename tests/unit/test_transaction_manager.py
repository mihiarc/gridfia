"""
Unit tests for the TransactionManager class.
"""
import pytest
import json
from pathlib import Path
from datetime import datetime
import time

import pandas as pd
from sqlalchemy import text

from processing.transaction_manager import TransactionManager

def test_transaction_creation(test_transaction_manager):
    """Test transaction initialization."""
    with test_transaction_manager.transaction("test_table") as engine:
        # Create a test table
        with engine.connect() as conn:
            conn.execute(text("""
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY,
                value TEXT
            );
            """))
            conn.commit()
    
    # Check transaction was created and completed
    assert len(test_transaction_manager.active_transactions) == 0

def test_backup_creation(test_transaction_manager):
    """Test backup table creation."""
    table_name = "test_backup"
    
    with test_transaction_manager.transaction(table_name) as engine:
        # Create and populate test table
        with engine.connect() as conn:
            conn.execute(text(f"""
            CREATE TABLE {table_name} (
                id INTEGER PRIMARY KEY,
                value TEXT
            );
            """))
            conn.execute(text(f"""
            INSERT INTO {table_name} (id, value)
            VALUES (1, 'test');
            """))
            conn.commit()
    
    # Check backup was created and cleaned up
    with engine.connect() as conn:
        result = conn.execute(text("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name LIKE 'test_backup%';
        """))
        backup_tables = result.fetchall()
    
    assert len(backup_tables) == 0  # Backup should be cleaned up

def test_checkpoint_creation(test_transaction_manager, test_data_dir):
    """Test checkpoint file creation."""
    table_name = "test_checkpoint"
    
    with test_transaction_manager.transaction(table_name) as engine:
        # Create test table
        with engine.connect() as conn:
            conn.execute(text(f"""
            CREATE TABLE {table_name} (
                id INTEGER PRIMARY KEY,
                value TEXT
            );
            """))
            conn.commit()
    
    # Check checkpoint files
    checkpoint_files = list(test_data_dir.glob("checkpoints/*.json"))
    assert len(checkpoint_files) > 0
    
    # Verify checkpoint content
    with open(checkpoint_files[0]) as f:
        checkpoint = json.load(f)
        assert checkpoint["table_name"] == table_name
        assert "timestamp" in checkpoint
        assert "state" in checkpoint

def test_rollback_procedure(test_transaction_manager):
    """Test rollback functionality."""
    table_name = "test_rollback"
    
    # Create initial table
    with test_transaction_manager.transaction(table_name) as engine:
        with engine.connect() as conn:
            conn.execute(text(f"""
            CREATE TABLE {table_name} (
                id INTEGER PRIMARY KEY,
                value TEXT
            );
            """))
            conn.execute(text(f"""
            INSERT INTO {table_name} (id, value)
            VALUES (1, 'initial');
            """))
            conn.commit()
    
    # Try modification that will fail
    try:
        with test_transaction_manager.transaction(table_name) as engine:
            with engine.connect() as conn:
                conn.execute(text(f"""
                INSERT INTO {table_name} (id, value)
                VALUES (1, 'duplicate');  -- Should fail (duplicate key)
                """))
                conn.commit()
    except:
        pass
    
    # Verify original data was restored
    with test_transaction_manager.engine.connect() as conn:
        result = conn.execute(text(f"""
        SELECT value FROM {table_name} WHERE id = 1;
        """))
        value = result.fetchone()[0]
    
    assert value == 'initial'

def test_transaction_status(test_transaction_manager):
    """Test transaction status tracking."""
    table_name = "test_status"
    
    with test_transaction_manager.transaction(table_name) as engine:
        # Check active transaction
        active = test_transaction_manager.list_active_transactions()
        assert len(active) == 1
        assert active[0]["table_name"] == table_name
        assert "start_time" in active[0]
        
        # Get specific transaction status
        transaction_id = list(test_transaction_manager.active_transactions.keys())[0]
        status = test_transaction_manager.get_transaction_status(transaction_id)
        assert status is not None
        assert status["table_name"] == table_name
        assert status["schema"] == "public"

def test_concurrent_transactions(test_transaction_manager):
    """Test handling of concurrent transactions."""
    table1 = "test_concurrent_1"
    table2 = "test_concurrent_2"
    
    # Start two transactions
    with test_transaction_manager.transaction(table1) as engine1:
        with test_transaction_manager.transaction(table2) as engine2:
            # Check both transactions are active
            active = test_transaction_manager.list_active_transactions()
            assert len(active) == 2
            assert {t["table_name"] for t in active} == {table1, table2}

def test_transaction_isolation(test_transaction_manager):
    """Test transaction isolation levels."""
    table_name = "test_isolation"
    
    # Create table in first transaction
    with test_transaction_manager.transaction(table_name) as engine:
        with engine.connect() as conn:
            conn.execute(text(f"""
            CREATE TABLE {table_name} (
                id INTEGER PRIMARY KEY,
                value TEXT
            );
            """))
            conn.commit()
    
    # Modify in second transaction (should fail)
    try:
        with test_transaction_manager.transaction(table_name) as engine:
            with engine.connect() as conn:
                conn.execute(text(f"""
                DROP TABLE {table_name};
                """))
                conn.commit()
    except:
        pass
    
    # Verify table still exists
    with test_transaction_manager.engine.connect() as conn:
        result = conn.execute(text("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name = :name;
        """), {"name": table_name})
        assert result.fetchone() is not None

def test_checkpoint_recovery(test_transaction_manager, test_data_dir):
    """Test recovery from checkpoints."""
    table_name = "test_recovery"
    
    # Start transaction that will be interrupted
    try:
        with test_transaction_manager.transaction(table_name) as engine:
            with engine.connect() as conn:
                conn.execute(text(f"""
                CREATE TABLE {table_name} (
                    id INTEGER PRIMARY KEY,
                    value TEXT
                );
                """))
                conn.commit()
            raise Exception("Simulated failure")
    except:
        pass
    
    # Check checkpoint was created
    checkpoint_files = list(test_data_dir.glob("checkpoints/*.json"))
    assert len(checkpoint_files) > 0
    
    # Verify checkpoint contains error info
    with open(checkpoint_files[0]) as f:
        checkpoint = json.load(f)
        assert "error" in checkpoint["state"]
        assert "error_time" in checkpoint["state"]

def test_backup_cleanup(test_transaction_manager):
    """Test cleanup of backup tables."""
    table_name = "test_cleanup"
    
    # Create and complete transaction
    with test_transaction_manager.transaction(table_name) as engine:
        with engine.connect() as conn:
            conn.execute(text(f"""
            CREATE TABLE {table_name} (
                id INTEGER PRIMARY KEY,
                value TEXT
            );
            """))
            conn.commit()
    
    # Verify no backup tables remain
    with test_transaction_manager.engine.connect() as conn:
        result = conn.execute(text("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name LIKE 'test_cleanup_backup%';
        """))
        assert len(result.fetchall()) == 0

def test_transaction_timing(test_transaction_manager):
    """Test transaction timing tracking."""
    table_name = "test_timing"
    
    start_time = datetime.now()
    with test_transaction_manager.transaction(table_name) as engine:
        time.sleep(0.1)  # Simulate work
    end_time = datetime.now()
    
    # Get last checkpoint
    checkpoint_files = sorted(test_transaction_manager.checkpoint_dir.glob("*.json"))
    with open(checkpoint_files[-1]) as f:
        checkpoint = json.load(f)
    
    # Verify timing information
    transaction_time = datetime.fromisoformat(checkpoint["timestamp"])
    assert start_time <= transaction_time <= end_time

def test_error_checkpoint_format(test_transaction_manager):
    """Test error checkpoint format."""
    table_name = "test_error_format"
    error_message = "Test error message"
    
    try:
        with test_transaction_manager.transaction(table_name) as engine:
            raise ValueError(error_message)
    except ValueError:
        pass
    
    # Get error checkpoint
    checkpoint_files = sorted(test_transaction_manager.checkpoint_dir.glob("*.json"))
    with open(checkpoint_files[-1]) as f:
        checkpoint = json.load(f)
    
    # Verify error information
    assert "error" in checkpoint["state"]
    assert checkpoint["state"]["error"] == error_message
    assert "error_time" in checkpoint["state"] 