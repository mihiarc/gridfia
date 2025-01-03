"""Development environment tests for the heirs-property project."""

import os
import sys
import json
import pytest
import subprocess
from pathlib import Path
from typing import Dict

@pytest.fixture
def vscode_settings(project_root: Path) -> Dict:
    """Load VSCode settings."""
    settings_file = project_root / ".vscode" / "settings.json"
    assert settings_file.exists(), "VSCode settings file not found"
    return json.loads(settings_file.read_text())

@pytest.fixture
def python_path() -> str:
    """Get Python interpreter path."""
    return sys.executable

def test_vscode_python_settings(vscode_settings: Dict):
    """Test VSCode Python settings."""
    # Check Python settings
    assert "python.defaultInterpreterPath" in vscode_settings
    assert "python.analysis.typeCheckingMode" in vscode_settings
    assert "python.formatting.provider" in vscode_settings
    assert "python.linting.enabled" in vscode_settings
    
    # Check formatting settings
    assert vscode_settings["python.formatting.provider"] == "ruff"
    assert vscode_settings["python.linting.enabled"] is True

def test_vscode_test_settings(vscode_settings: Dict):
    """Test VSCode test configuration."""
    assert "python.testing.pytestEnabled" in vscode_settings
    assert vscode_settings["python.testing.pytestEnabled"] is True
    assert "python.testing.unittestEnabled" in vscode_settings
    assert vscode_settings["python.testing.unittestEnabled"] is False

def test_python_environment(python_path: str):
    """Test Python environment configuration."""
    # Check Python version
    version = sys.version_info
    assert version.major == 3
    assert version.minor >= 9, "Python version should be 3.9 or higher"
    
    # Check virtual environment
    assert "VIRTUAL_ENV" in os.environ, "Not running in a virtual environment"
    
    # Check if we can import required packages
    required_packages = [
        "numpy",
        "pandas",
        "geopandas",
        "rasterio",
        "shapely",
        "pytest",
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError as e:
            pytest.fail(f"Failed to import {package}: {e}")

def test_ruff_configuration(project_root: Path):
    """Test Ruff configuration."""
    # Check if ruff config exists in pyproject.toml
    pyproject_file = project_root / "pyproject.toml"
    assert pyproject_file.exists(), "pyproject.toml not found"
    
    # Run ruff check
    result = subprocess.run(
        ["ruff", "check", "."],
        capture_output=True,
        text=True,
        cwd=project_root
    )
    
    # Print output for debugging
    if result.returncode != 0:
        print("Ruff output:", result.stdout)
        print("Ruff errors:", result.stderr)
    
    assert result.returncode == 0, "Ruff check failed"

def test_mypy_configuration(project_root: Path):
    """Test MyPy configuration."""
    # Check if mypy config exists in pyproject.toml
    pyproject_file = project_root / "pyproject.toml"
    assert pyproject_file.exists(), "pyproject.toml not found"
    
    # Run mypy check
    result = subprocess.run(
        ["mypy", "src"],
        capture_output=True,
        text=True,
        cwd=project_root
    )
    
    # Print output for debugging
    if result.returncode != 0:
        print("MyPy output:", result.stdout)
        print("MyPy errors:", result.stderr)
    
    assert result.returncode == 0, "MyPy check failed"

def test_pre_commit_hooks(project_root: Path):
    """Test pre-commit hooks configuration."""
    # Check if pre-commit config exists
    config_file = project_root / ".pre-commit-config.yaml"
    assert config_file.exists(), "pre-commit config not found"
    
    # Run pre-commit check
    result = subprocess.run(
        ["pre-commit", "run", "--all-files"],
        capture_output=True,
        text=True,
        cwd=project_root
    )
    
    # Print output for debugging
    if result.returncode != 0:
        print("pre-commit output:", result.stdout)
        print("pre-commit errors:", result.stderr)
    
    assert result.returncode == 0, "pre-commit checks failed"

def test_git_hooks(project_root: Path):
    """Test Git hooks installation."""
    git_hooks_dir = project_root / ".git" / "hooks"
    assert git_hooks_dir.exists(), "Git hooks directory not found"
    
    # Check if pre-commit hook is installed
    pre_commit_hook = git_hooks_dir / "pre-commit"
    assert pre_commit_hook.exists(), "pre-commit hook not installed"
    assert os.access(pre_commit_hook, os.X_OK), "pre-commit hook not executable"

def test_docker_development(project_root: Path):
    """Test Docker development configuration."""
    # Check Docker files
    assert (project_root / "docker-compose.yml").exists(), "docker-compose.yml not found"
    assert (project_root / "Dockerfile.jupyter").exists(), "Dockerfile.jupyter not found"
    assert (project_root / "Dockerfile.processing").exists(), "Dockerfile.processing not found"
    
    # Check if Docker is running
    result = subprocess.run(
        ["docker", "info"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, "Docker not running"

def test_environment_variables(load_env):
    """Test environment variable configuration."""
    required_vars = [
        "RAW_DATA_PATH",
        "PROCESSED_DATA_PATH",
        "INTERIM_DATA_PATH",
        "DB_HOST",
        "DB_PORT",
        "DB_NAME",
        "DB_USER",
        "DB_PASSWORD",
    ]
    
    for var in required_vars:
        assert var in load_env, f"Missing environment variable: {var}"
        assert load_env[var] is not None, f"Environment variable not set: {var}"

def test_project_structure(project_root: Path):
    """Test project directory structure."""
    required_dirs = [
        "src",
        "tests",
        "docs",
        "notebooks",
        "data",
        "config",
        "logs",
    ]
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        assert dir_path.exists(), f"Required directory missing: {dir_name}"
        assert dir_path.is_dir(), f"Not a directory: {dir_name}" 