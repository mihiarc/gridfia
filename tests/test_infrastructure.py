"""Infrastructure tests for the heirs-property project."""

import os
import time
import docker
import pytest
import requests
from pathlib import Path
from typing import Generator

@pytest.fixture(scope="session")
def docker_client() -> docker.DockerClient:
    """Create a Docker client."""
    return docker.from_env()

@pytest.fixture(scope="session")
def project_containers(docker_client: docker.DockerClient) -> Generator[list, None, None]:
    """Get all project containers."""
    containers = docker_client.containers.list(
        filters={"label": "com.docker.compose.project=heirs-property"}
    )
    yield containers

def test_container_health(project_containers: list):
    """Test if all containers are healthy."""
    for container in project_containers:
        assert container.status == "running"
        
        # Check container health if health check is configured
        if hasattr(container, "health"):
            assert container.health["status"] == "healthy"

def test_container_logs(project_containers: list):
    """Test if containers are logging properly."""
    for container in project_containers:
        logs = container.logs(tail=100).decode("utf-8")
        assert logs, f"No logs found for container {container.name}"
        assert "error" not in logs.lower(), f"Errors found in {container.name} logs"

def test_volume_persistence(docker_client: docker.DockerClient):
    """Test data persistence in volumes."""
    # Get PostGIS container
    postgis = docker_client.containers.get("heirs-property-postgis-1")
    
    # Create a test table
    postgis.exec_run(
        "psql -U postgres -d heirs_property -c 'CREATE TABLE IF NOT EXISTS test_persistence (id serial PRIMARY KEY);'"
    )
    
    # Restart container
    postgis.restart()
    time.sleep(5)  # Wait for container to restart
    
    # Check if table still exists
    result = postgis.exec_run(
        "psql -U postgres -d heirs_property -c '\dt test_persistence;'"
    )
    assert result.exit_code == 0, "Table not persisted after restart"

def test_network_connectivity(project_containers: list):
    """Test network connectivity between containers."""
    # Get PostGIS container
    postgis = next(c for c in project_containers if "postgis" in c.name)
    
    # Test connection from processing container
    processing = next(c for c in project_containers if "processing" in c.name)
    result = processing.exec_run(
        f"pg_isready -h postgis -U postgres"
    )
    assert result.exit_code == 0, "Processing container cannot connect to PostGIS"

def test_jupyter_accessibility():
    """Test if Jupyter is accessible."""
    response = requests.get("http://localhost:8888")
    assert response.status_code == 200, "Jupyter not accessible"

def test_resource_limits(project_containers: list):
    """Test container resource limits."""
    for container in project_containers:
        stats = container.stats(stream=False)
        
        # Check memory usage
        memory_usage = stats["memory_stats"]["usage"]
        memory_limit = stats["memory_stats"]["limit"]
        memory_percent = (memory_usage / memory_limit) * 100
        
        assert memory_percent < 80, f"High memory usage in {container.name}: {memory_percent}%"
        
        # Check CPU usage
        cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - \
                   stats["precpu_stats"]["cpu_usage"]["total_usage"]
        system_delta = stats["cpu_stats"]["system_cpu_usage"] - \
                      stats["precpu_stats"]["system_cpu_usage"]
        cpu_percent = (cpu_delta / system_delta) * 100
        
        assert cpu_percent < 80, f"High CPU usage in {container.name}: {cpu_percent}%"

def test_log_rotation():
    """Test log rotation configuration."""
    log_dir = Path("logs")
    assert log_dir.exists(), "Log directory not found"
    
    # Check PostGIS logs
    postgis_logs = log_dir / "postgis"
    assert postgis_logs.exists(), "PostGIS log directory not found"
    
    # Ensure log files are not too large
    for log_file in postgis_logs.glob("*.log"):
        size_mb = log_file.stat().st_size / (1024 * 1024)
        assert size_mb < 100, f"Log file {log_file} is too large: {size_mb}MB"

def test_backup_directories():
    """Test backup directory structure."""
    backup_dir = Path("backup/postgres")
    assert backup_dir.exists(), "Backup directory not found"
    assert os.access(backup_dir, os.W_OK), "Backup directory not writable"

def test_data_directories():
    """Test data directory structure."""
    data_dirs = ["raw", "processed", "interim"]
    for dir_name in data_dirs:
        data_dir = Path(f"data/{dir_name}")
        assert data_dir.exists(), f"Data directory {dir_name} not found"
        assert os.access(data_dir, os.W_OK), f"Data directory {dir_name} not writable" 