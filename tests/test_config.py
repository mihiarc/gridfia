"""
Test configuration loading and validation.
"""

import pytest
from pathlib import Path
from unittest.mock import patch
from config import PathConfig

def test_external_drive():
    """Test external drive validation."""
    config_path = Path("config/paths.yaml")
    path_config = PathConfig.from_yaml(config_path)
    
    # Check if external drive is mounted
    is_mounted = path_config.input.check_external_drive()
    print(f"\nExternal drive mounted: {is_mounted}")
    
    if not is_mounted:
        pytest.skip("External drive not mounted - skipping further tests")

def test_load_config():
    """Test loading configuration from YAML."""
    # Load the configuration
    config_path = Path("config/paths.yaml")
    path_config = PathConfig.from_yaml(config_path)
    
    # Print configuration for inspection
    print("\nLoaded configuration:")
    print(path_config.input)
    
    # Test that paths are absolute
    assert path_config.input.ndvi_dir.is_absolute()
    assert path_config.input.property_file.is_absolute()
    assert path_config.input.nc_parcels_file.is_absolute()
    assert path_config.input.auxiliary_data['county_boundaries'].is_absolute()
    
    # Test output directories were created
    assert path_config.output.processed_dir.exists()
    assert path_config.output.results_dir.exists()
    assert path_config.output.reports_dir.exists()
    assert path_config.output.figures_dir.exists()
    
    # Test temp directory
    assert path_config.temp.root_dir.exists()

@patch('pathlib.Path.exists')
def test_validate_paths(mock_exists):
    """Test path validation."""
    # Mock all paths to exist
    mock_exists.return_value = True
    
    config_path = Path("config/paths.yaml")
    path_config = PathConfig.from_yaml(config_path)
    
    # This should not raise any exceptions since we mocked the paths to exist
    path_config.input.validate_paths()

def test_validate_formats():
    """Test file format validation."""
    config_path = Path("config/paths.yaml")
    path_config = PathConfig.from_yaml(config_path)
    
    # This should not raise any exceptions
    path_config.input.validate_formats()

def test_path_types():
    """Test that all paths are Path objects."""
    config_path = Path("config/paths.yaml")
    path_config = PathConfig.from_yaml(config_path)
    
    assert isinstance(path_config.input.ndvi_dir, Path)
    assert isinstance(path_config.input.property_file, Path)
    assert isinstance(path_config.input.nc_parcels_file, Path)
    assert all(isinstance(p, Path) for p in path_config.input.auxiliary_data.values())

if __name__ == "__main__":
    print("\nTesting Configuration System")
    print("===========================")
    
    # First check external drive
    print("\nChecking external drive...")
    test_external_drive()
    
    # Load and display configuration
    config_path = Path("config/paths.yaml")
    path_config = PathConfig.from_yaml(config_path)
    print("\nCurrent Path Configuration:")
    print(path_config.input)
    
    print("\nRunning validation tests...")
    try:
        test_validate_formats()
        print("✓ Format validation passed")
        
        test_validate_paths()
        print("✓ Path validation passed")
        
        test_path_types()
        print("✓ Path type validation passed")
        
        test_load_config()
        print("✓ Configuration loading passed")
        
        print("\nAll tests passed successfully!")
        
    except ValueError as e:
        print(f"\n❌ Validation Error: {e}")
    except Exception as e:
        print(f"\n❌ Test Error: {e}") 