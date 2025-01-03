"""Pytest configuration and shared fixtures."""

import os
import pytest
import docker
import psycopg2
from pathlib import Path

# Environment setup
os.environ.setdefault("POSTGRES_DB", "heirs_property")
os.environ.setdefault("POSTGRES_USER", "heirs_user")
os.environ.setdefault("POSTGRES_PASSWORD", "dev_password_123")
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_PORT", "5432")

@pytest.fixture(scope="session")
def docker_client():
    """Create a Docker client for container management."""
    return docker.from_env()

@pytest.fixture(scope="session")
def db_params():
    """Database connection parameters."""
    return {
        "dbname": os.getenv("POSTGRES_DB"),
        "user": os.getenv("POSTGRES_USER"),
        "password": os.getenv("POSTGRES_PASSWORD"),
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT")
    }

@pytest.fixture(scope="session")
def db_connection(db_params):
    """Create a database connection for testing."""
    conn = psycopg2.connect(**db_params)
    yield conn
    conn.close()

@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent

@pytest.fixture(scope="session")
def data_dir(project_root):
    """Return the data directory."""
    return project_root / "data"

@pytest.fixture(scope="session")
def logs_dir(project_root):
    """Return the logs directory."""
    return project_root / "logs"

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle markers."""
    if config.getoption("--skip-integration", default=False):
        skip_integration = pytest.mark.skip(reason="Integration tests skipped")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)

def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--skip-integration",
        action="store_true",
        default=False,
        help="Skip integration tests"
    ) 