"""Integration tests for the Heirs Property system."""

import os
import pytest
import psycopg2
import pandas as pd
from pathlib import Path
from datetime import datetime
import time

@pytest.fixture(scope="module")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory for test data."""
    test_dir = tmp_path_factory.mktemp("test_data")
    return test_dir

@pytest.mark.integration
class TestContainerCommunication:
    """Test communication between containers."""
    
    def test_postgis_connection(self, db_connection):
        """Test PostgreSQL connection and PostGIS functionality."""
        with db_connection.cursor() as cur:
            # Test PostGIS version
            cur.execute("SELECT PostGIS_Version();")
            version = cur.fetchone()[0]
            assert version is not None
            assert "3.3" in version
            
            # Test spatial reference system
            cur.execute("SELECT srtext FROM spatial_ref_sys WHERE srid = 2264;")
            srid_info = cur.fetchone()[0]
            assert "NAD83 / North Carolina" in srid_info
    
    def test_jupyter_db_connection(self, db_connection):
        """Test JupyterLab's ability to connect to PostgreSQL."""
        from sqlalchemy import create_engine
        
        try:
            # Create SQLAlchemy engine
            engine = create_engine(
                f'postgresql://{os.getenv("POSTGRES_USER")}:{os.getenv("POSTGRES_PASSWORD")}@localhost:5432/{os.getenv("POSTGRES_DB")}'
            )
            
            # Create a test table
            with db_connection.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS test_jupyter (
                        id SERIAL PRIMARY KEY,
                        test_data TEXT
                    );
                """)
                db_connection.commit()
            
            # Use pandas to interact with the database
            df = pd.DataFrame({'test_data': ['test1', 'test2']})
            df.to_sql('test_jupyter', engine, if_exists='append', index=False, schema='public')
            
            # Verify data
            result_df = pd.read_sql("SELECT * FROM test_jupyter", engine)
            assert len(result_df) >= 2
            assert 'test_data' in result_df.columns
            
        finally:
            # Cleanup
            with db_connection.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS test_jupyter;")
                db_connection.commit()

    def test_log_persistence(self, docker_client):
        """Test if logs are properly persisted."""
        log_dir = Path("logs/postgis")
        assert log_dir.exists(), "Log directory not found"
        
        # Generate some database activity to create logs
        with psycopg2.connect(
            dbname=os.getenv("POSTGRES_DB", "heirs_property"),
            user=os.getenv("POSTGRES_USER", "heirs_user"),
            password=os.getenv("POSTGRES_PASSWORD", "dev_password_123"),
            host="localhost",
            port=5432
        ) as conn:
            with conn.cursor() as cur:
                # Execute some queries to generate log entries
                cur.execute("SELECT PostGIS_Version();")
                cur.execute("SELECT current_timestamp;")
                cur.execute("SELECT current_database();")
        
        # Wait briefly for logs to be written
        time.sleep(2)
        
        # Check if log files exist in the container
        containers = docker_client.containers.list(
            filters={"name": "heirs-property_postgis"}
        )
        if not containers:
            pytest.skip("PostgreSQL container not found")
            
        postgis_container = containers[0]
        
        # Check PostgreSQL logs in the container
        exec_result = postgis_container.exec_run("ls -l /var/log/postgresql")
        assert exec_result.exit_code == 0
        assert len(exec_result.output.decode().strip().split("\n")) > 0

class TestVolumePersistence:
    """Test volume persistence across container restarts."""
    
    def test_data_persistence(self, docker_client, test_data_dir):
        """Test if data persists after container restart."""
        # Create test data
        test_file = test_data_dir / "test_data.txt"
        test_content = f"Test data created at {datetime.now()}"
        test_file.write_text(test_content)
        
        # Get PostgreSQL container
        containers = docker_client.containers.list(
            filters={"name": "heirs-property_postgis"}
        )
        if not containers:
            pytest.skip("PostgreSQL container not found")
        
        postgis_container = containers[0]
        
        # Copy test data to container
        with test_file.open('rb') as f:
            postgis_container.put_archive('/test_data', f.read())
        
        # Stop container
        postgis_container.stop()
        
        # Start container
        postgis_container.start()
        
        # Wait for container to be healthy
        time.sleep(5)
        
        # Verify data persistence
        exec_result = postgis_container.exec_run("cat /test_data/test_data.txt")
        assert exec_result.exit_code == 0
        assert test_content in exec_result.output.decode()

    def test_log_persistence(self, docker_client):
        """Test if logs are properly persisted."""
        log_dir = Path("logs/postgis")
        assert log_dir.exists(), "Log directory not found"
        
        # Check if log files are being created
        log_files = list(log_dir.glob("*.log"))
        assert len(log_files) > 0, "No log files found"
        
        # Verify log file is being written to
        latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
        initial_size = latest_log.stat().st_size
        
        # Generate some database activity
        with psycopg2.connect(
            dbname=os.getenv("POSTGRES_DB", "heirs_property"),
            user=os.getenv("POSTGRES_USER", "heirs_user"),
            password=os.getenv("POSTGRES_PASSWORD", "dev_password_123"),
            host="localhost",
            port=5432
        ) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT PostGIS_Version();")
        
        # Verify log size has increased
        new_size = latest_log.stat().st_size
        assert new_size > initial_size, "Log file not updated" 