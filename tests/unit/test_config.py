"""
Comprehensive tests for GridFIA configuration module.

This module provides comprehensive test coverage for the configuration system
including VisualizationConfig, ProcessingConfig, CalculationConfig, GridFIASettings,
and the load_settings/save_settings utility functions.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest
import yaml

from gridfia.config import (
    VisualizationConfig,
    ProcessingConfig,
    CalculationConfig,
    GridFIASettings,
    BigMapSettings,
    OutputFormat,
    load_settings,
    save_settings,
    settings,
)


# =============================================================================
# VisualizationConfig Tests
# =============================================================================


class TestVisualizationConfig:
    """Test VisualizationConfig model validation and defaults."""

    def test_default_values(self):
        """Test that default values are correctly set."""
        config = VisualizationConfig()

        assert config.default_dpi == 300
        assert config.default_figure_size == (16, 12)
        assert config.font_size == 12
        assert config.color_maps == {
            "biomass": "viridis",
            "diversity": "plasma",
            "richness": "Spectral_r"
        }

    def test_valid_dpi_values(self):
        """Test DPI values within valid range (72-600)."""
        # Minimum valid DPI
        config_min = VisualizationConfig(default_dpi=72)
        assert config_min.default_dpi == 72

        # Maximum valid DPI
        config_max = VisualizationConfig(default_dpi=600)
        assert config_max.default_dpi == 600

        # Common DPI values
        for dpi in [96, 150, 300, 600]:
            config = VisualizationConfig(default_dpi=dpi)
            assert config.default_dpi == dpi

    def test_dpi_too_low(self):
        """Test that DPI below 72 raises validation error."""
        with pytest.raises(ValueError, match="greater than or equal to 72"):
            VisualizationConfig(default_dpi=71)

        with pytest.raises(ValueError, match="greater than or equal to 72"):
            VisualizationConfig(default_dpi=0)

        with pytest.raises(ValueError, match="greater than or equal to 72"):
            VisualizationConfig(default_dpi=-100)

    def test_dpi_too_high(self):
        """Test that DPI above 600 raises validation error."""
        with pytest.raises(ValueError, match="less than or equal to 600"):
            VisualizationConfig(default_dpi=601)

        with pytest.raises(ValueError, match="less than or equal to 600"):
            VisualizationConfig(default_dpi=1200)

    def test_valid_font_size_values(self):
        """Test font size values within valid range (8-24)."""
        # Minimum valid font size
        config_min = VisualizationConfig(font_size=8)
        assert config_min.font_size == 8

        # Maximum valid font size
        config_max = VisualizationConfig(font_size=24)
        assert config_max.font_size == 24

        # Common font sizes
        for size in [10, 12, 14, 16, 18]:
            config = VisualizationConfig(font_size=size)
            assert config.font_size == size

    def test_font_size_too_low(self):
        """Test that font size below 8 raises validation error."""
        with pytest.raises(ValueError, match="greater than or equal to 8"):
            VisualizationConfig(font_size=7)

        with pytest.raises(ValueError, match="greater than or equal to 8"):
            VisualizationConfig(font_size=0)

        with pytest.raises(ValueError, match="greater than or equal to 8"):
            VisualizationConfig(font_size=-5)

    def test_font_size_too_high(self):
        """Test that font size above 24 raises validation error."""
        with pytest.raises(ValueError, match="less than or equal to 24"):
            VisualizationConfig(font_size=25)

        with pytest.raises(ValueError, match="less than or equal to 24"):
            VisualizationConfig(font_size=100)

    def test_custom_figure_size(self):
        """Test custom figure size tuple."""
        config = VisualizationConfig(default_figure_size=(10.5, 8.0))
        assert config.default_figure_size == (10.5, 8.0)

        config = VisualizationConfig(default_figure_size=(20, 15))
        assert config.default_figure_size == (20, 15)

    def test_custom_color_maps(self):
        """Test custom color map dictionary."""
        custom_maps = {
            "biomass": "plasma",
            "diversity": "viridis",
            "richness": "coolwarm",
            "custom": "twilight"
        }
        config = VisualizationConfig(color_maps=custom_maps)
        assert config.color_maps == custom_maps

    def test_empty_color_maps(self):
        """Test that empty color maps dict is allowed."""
        config = VisualizationConfig(color_maps={})
        assert config.color_maps == {}


# =============================================================================
# OutputFormat Enum Tests
# =============================================================================


class TestOutputFormat:
    """Test OutputFormat enum."""

    def test_enum_values(self):
        """Test that OutputFormat enum has expected values."""
        assert OutputFormat.GEOTIFF.value == "geotiff"
        assert OutputFormat.ZARR.value == "zarr"
        assert OutputFormat.NETCDF.value == "netcdf"

    def test_enum_is_string_subclass(self):
        """Test that OutputFormat is a string subclass."""
        assert isinstance(OutputFormat.GEOTIFF, str)
        assert OutputFormat.GEOTIFF == "geotiff"

    def test_enum_members(self):
        """Test all enum members."""
        members = list(OutputFormat)
        assert len(members) == 3
        assert OutputFormat.GEOTIFF in members
        assert OutputFormat.ZARR in members
        assert OutputFormat.NETCDF in members

    def test_enum_from_string(self):
        """Test creating enum from string."""
        assert OutputFormat("geotiff") == OutputFormat.GEOTIFF
        assert OutputFormat("zarr") == OutputFormat.ZARR
        assert OutputFormat("netcdf") == OutputFormat.NETCDF

    def test_enum_invalid_string_raises_error(self):
        """Test that invalid string raises ValueError."""
        with pytest.raises(ValueError):
            OutputFormat("invalid")

        with pytest.raises(ValueError):
            OutputFormat("csv")


# =============================================================================
# ProcessingConfig Tests
# =============================================================================


class TestProcessingConfig:
    """Test ProcessingConfig model validation and defaults."""

    def test_default_values(self):
        """Test that default values are correctly set."""
        config = ProcessingConfig()

        assert config.max_workers is None
        assert config.memory_limit_gb == 8.0
        assert config.temp_dir is None

    def test_custom_max_workers(self):
        """Test custom max_workers value."""
        config = ProcessingConfig(max_workers=4)
        assert config.max_workers == 4

        config = ProcessingConfig(max_workers=1)
        assert config.max_workers == 1

        config = ProcessingConfig(max_workers=128)
        assert config.max_workers == 128

    def test_valid_memory_limit(self):
        """Test valid memory limit values."""
        config = ProcessingConfig(memory_limit_gb=0.5)
        assert config.memory_limit_gb == 0.5

        config = ProcessingConfig(memory_limit_gb=16.0)
        assert config.memory_limit_gb == 16.0

        config = ProcessingConfig(memory_limit_gb=128.0)
        assert config.memory_limit_gb == 128.0

    def test_memory_limit_must_be_positive(self):
        """Test that memory limit must be greater than 0."""
        with pytest.raises(ValueError, match="greater than 0"):
            ProcessingConfig(memory_limit_gb=0)

        with pytest.raises(ValueError, match="greater than 0"):
            ProcessingConfig(memory_limit_gb=-1)

        with pytest.raises(ValueError, match="greater than 0"):
            ProcessingConfig(memory_limit_gb=-8.0)

    def test_temp_dir_none(self):
        """Test that None temp_dir is valid."""
        config = ProcessingConfig(temp_dir=None)
        assert config.temp_dir is None

    def test_temp_dir_existing_path(self, tmp_path: Path):
        """Test temp_dir with existing directory."""
        existing_dir = tmp_path / "existing_temp"
        existing_dir.mkdir()

        config = ProcessingConfig(temp_dir=existing_dir)
        assert config.temp_dir == existing_dir
        assert config.temp_dir.exists()

    def test_temp_dir_creates_missing_directory(self, tmp_path: Path):
        """Test that validator creates missing temp directory."""
        new_dir = tmp_path / "new_temp_dir"
        assert not new_dir.exists()

        config = ProcessingConfig(temp_dir=new_dir)
        assert config.temp_dir == new_dir
        assert new_dir.exists()

    def test_temp_dir_creates_nested_directory(self, tmp_path: Path):
        """Test that validator creates nested temp directory."""
        nested_dir = tmp_path / "deep" / "nested" / "temp"
        assert not nested_dir.exists()

        config = ProcessingConfig(temp_dir=nested_dir)
        assert config.temp_dir == nested_dir
        assert nested_dir.exists()

    def test_temp_dir_string_path_converted(self, tmp_path: Path):
        """Test that string path is converted to Path object."""
        dir_path = tmp_path / "string_temp"
        config = ProcessingConfig(temp_dir=str(dir_path))

        assert isinstance(config.temp_dir, Path)
        assert config.temp_dir == dir_path
        assert dir_path.exists()

    def test_temp_dir_invalid_path_raises_error(self):
        """Test that invalid path raises validation error."""
        # Try to create a directory in a non-existent root on Unix
        # This test might behave differently on different OS
        invalid_path = Path("/nonexistent_root_xyz/temp_dir")

        # Skip this test if we somehow have root access
        if invalid_path.parent.exists():
            pytest.skip("Test requires non-existent parent directory")

        with pytest.raises(ValueError, match="Cannot create temp directory"):
            ProcessingConfig(temp_dir=invalid_path)


# =============================================================================
# CalculationConfig Tests
# =============================================================================


class TestCalculationConfig:
    """Test CalculationConfig model validation and defaults."""

    def test_required_name_field(self):
        """Test that name field is required."""
        with pytest.raises(ValueError, match="name"):
            CalculationConfig()

    def test_default_values(self):
        """Test that default values are correctly set."""
        config = CalculationConfig(name="test_calc")

        assert config.name == "test_calc"
        assert config.enabled is True
        assert config.parameters == {}
        assert config.output_format == OutputFormat.GEOTIFF
        assert config.output_name is None

    def test_all_fields(self):
        """Test setting all fields."""
        config = CalculationConfig(
            name="species_richness",
            enabled=False,
            parameters={"threshold": 0.5, "min_biomass": 10},
            output_format="zarr",
            output_name="custom_output"
        )

        assert config.name == "species_richness"
        assert config.enabled is False
        assert config.parameters == {"threshold": 0.5, "min_biomass": 10}
        assert config.output_format == OutputFormat.ZARR
        assert config.output_name == "custom_output"

    def test_various_calculation_names(self):
        """Test various calculation names."""
        names = [
            "species_richness",
            "shannon_diversity",
            "simpson_diversity",
            "total_biomass",
            "dominant_species",
            "custom-calc",
            "calc123"
        ]

        for name in names:
            config = CalculationConfig(name=name)
            assert config.name == name

    def test_complex_parameters(self):
        """Test complex nested parameters."""
        complex_params = {
            "threshold": 0.5,
            "options": {
                "normalize": True,
                "method": "fast"
            },
            "weights": [0.1, 0.2, 0.3, 0.4],
            "metadata": {
                "author": "test",
                "version": 1
            }
        }

        config = CalculationConfig(name="complex", parameters=complex_params)
        assert config.parameters == complex_params

    def test_output_format_valid_values(self):
        """Test valid output format enum values."""
        # Test with string values
        config = CalculationConfig(name="test", output_format="geotiff")
        assert config.output_format == OutputFormat.GEOTIFF

        config = CalculationConfig(name="test", output_format="zarr")
        assert config.output_format == OutputFormat.ZARR

        config = CalculationConfig(name="test", output_format="netcdf")
        assert config.output_format == OutputFormat.NETCDF

        # Test with enum values directly
        config = CalculationConfig(name="test", output_format=OutputFormat.GEOTIFF)
        assert config.output_format == OutputFormat.GEOTIFF

    def test_output_format_invalid_values(self):
        """Test that invalid output format values raise error."""
        with pytest.raises(ValueError, match="Input should be"):
            CalculationConfig(name="test", output_format="csv")

        with pytest.raises(ValueError, match="Input should be"):
            CalculationConfig(name="test", output_format="custom")

        with pytest.raises(ValueError, match="Input should be"):
            CalculationConfig(name="test", output_format="invalid")


# =============================================================================
# GridFIASettings Tests
# =============================================================================


class TestGridFIASettings:
    """Test GridFIASettings model and environment variable loading."""

    def test_default_values(self, tmp_path: Path, monkeypatch):
        """Test that default values are correctly set."""
        # Change to tmp_path to avoid creating directories in cwd
        monkeypatch.chdir(tmp_path)

        settings_obj = GridFIASettings()

        assert settings_obj.app_name == "GridFIA"
        assert settings_obj.debug is False
        assert settings_obj.verbose is False
        assert settings_obj.data_dir == Path("data")
        assert settings_obj.output_dir == Path("output")
        assert settings_obj.cache_dir == Path(".cache")
        assert isinstance(settings_obj.visualization, VisualizationConfig)
        assert isinstance(settings_obj.processing, ProcessingConfig)
        assert isinstance(settings_obj.calculations, list)
        assert len(settings_obj.calculations) == 3  # Default calculations
        assert settings_obj.species_codes == []

    def test_directories_created(self, tmp_path: Path, monkeypatch):
        """Test that directory validators create directories."""
        monkeypatch.chdir(tmp_path)

        data_dir = tmp_path / "new_data"
        output_dir = tmp_path / "new_output"
        cache_dir = tmp_path / "new_cache"

        # None of these should exist yet
        assert not data_dir.exists()
        assert not output_dir.exists()
        assert not cache_dir.exists()

        settings_obj = GridFIASettings(
            data_dir=data_dir,
            output_dir=output_dir,
            cache_dir=cache_dir
        )

        # All should be created now
        assert data_dir.exists()
        assert output_dir.exists()
        assert cache_dir.exists()

    def test_nested_directories_created(self, tmp_path: Path, monkeypatch):
        """Test that nested directories are created."""
        monkeypatch.chdir(tmp_path)

        deep_dir = tmp_path / "a" / "b" / "c" / "data"

        settings_obj = GridFIASettings(data_dir=deep_dir)
        assert deep_dir.exists()

    def test_custom_nested_configs(self, tmp_path: Path, monkeypatch):
        """Test custom nested configuration objects."""
        monkeypatch.chdir(tmp_path)

        viz_config = VisualizationConfig(default_dpi=150, font_size=14)
        proc_config = ProcessingConfig(max_workers=4, memory_limit_gb=16.0)

        settings_obj = GridFIASettings(
            data_dir=tmp_path / "data",
            visualization=viz_config,
            processing=proc_config
        )

        assert settings_obj.visualization.default_dpi == 150
        assert settings_obj.visualization.font_size == 14
        assert settings_obj.processing.max_workers == 4
        assert settings_obj.processing.memory_limit_gb == 16.0

    def test_custom_calculations_list(self, tmp_path: Path, monkeypatch):
        """Test custom calculations list."""
        monkeypatch.chdir(tmp_path)

        calcs = [
            CalculationConfig(name="calc1", enabled=True),
            CalculationConfig(name="calc2", enabled=False),
            CalculationConfig(name="calc3", parameters={"key": "value"})
        ]

        settings_obj = GridFIASettings(
            data_dir=tmp_path / "data",
            calculations=calcs
        )

        assert len(settings_obj.calculations) == 3
        assert settings_obj.calculations[0].name == "calc1"
        assert settings_obj.calculations[1].enabled is False
        assert settings_obj.calculations[2].parameters == {"key": "value"}

    def test_species_codes(self, tmp_path: Path, monkeypatch):
        """Test species_codes list."""
        monkeypatch.chdir(tmp_path)

        codes = ["0131", "0202", "0068", "0122"]

        settings_obj = GridFIASettings(
            data_dir=tmp_path / "data",
            species_codes=codes
        )

        assert settings_obj.species_codes == codes

    def test_debug_and_verbose_flags(self, tmp_path: Path, monkeypatch):
        """Test debug and verbose flags."""
        monkeypatch.chdir(tmp_path)

        settings_obj = GridFIASettings(
            data_dir=tmp_path / "data",
            debug=True,
            verbose=True
        )

        assert settings_obj.debug is True
        assert settings_obj.verbose is True

    def test_environment_variable_loading(self, tmp_path: Path, monkeypatch):
        """Test loading settings from GRIDFIA_ prefixed environment variables."""
        monkeypatch.chdir(tmp_path)

        # Set environment variables with GRIDFIA_ prefix
        monkeypatch.setenv("GRIDFIA_DEBUG", "true")
        monkeypatch.setenv("GRIDFIA_VERBOSE", "true")
        monkeypatch.setenv("GRIDFIA_APP_NAME", "CustomGridFIA")

        settings_obj = GridFIASettings(data_dir=tmp_path / "data")

        assert settings_obj.debug is True
        assert settings_obj.verbose is True
        assert settings_obj.app_name == "CustomGridFIA"

    def test_environment_variable_case_insensitive(self, tmp_path: Path, monkeypatch):
        """Test that environment variables are case insensitive."""
        monkeypatch.chdir(tmp_path)

        # Use lowercase env var
        monkeypatch.setenv("gridfia_debug", "true")

        settings_obj = GridFIASettings(data_dir=tmp_path / "data")

        # Should work with case_sensitive=False
        assert settings_obj.debug is True

    def test_extra_fields_ignored(self, tmp_path: Path, monkeypatch):
        """Test that extra fields are ignored with extra='ignore'."""
        monkeypatch.chdir(tmp_path)

        # This should not raise an error due to extra="ignore"
        settings_obj = GridFIASettings(
            data_dir=tmp_path / "data",
            unknown_field="should_be_ignored",
            another_unknown=123
        )

        assert not hasattr(settings_obj, "unknown_field")
        assert not hasattr(settings_obj, "another_unknown")

    def test_get_output_path(self, tmp_path: Path, monkeypatch):
        """Test get_output_path method."""
        monkeypatch.chdir(tmp_path)

        output_dir = tmp_path / "output"
        settings_obj = GridFIASettings(
            data_dir=tmp_path / "data",
            output_dir=output_dir
        )

        result = settings_obj.get_output_path("test_file.tif")
        assert result == output_dir / "test_file.tif"

        result = settings_obj.get_output_path("subdir/nested_file.zarr")
        assert result == output_dir / "subdir/nested_file.zarr"

    def test_get_temp_path_uses_processing_temp_dir(self, tmp_path: Path, monkeypatch):
        """Test get_temp_path uses processing.temp_dir when set."""
        monkeypatch.chdir(tmp_path)

        temp_dir = tmp_path / "custom_temp"
        proc_config = ProcessingConfig(temp_dir=temp_dir)

        settings_obj = GridFIASettings(
            data_dir=tmp_path / "data",
            processing=proc_config
        )

        result = settings_obj.get_temp_path("temp_file.tmp")
        assert result == temp_dir / "temp_file.tmp"

    def test_get_temp_path_falls_back_to_cache_dir(self, tmp_path: Path, monkeypatch):
        """Test get_temp_path falls back to cache_dir when temp_dir is None."""
        monkeypatch.chdir(tmp_path)

        cache_dir = tmp_path / "cache"
        settings_obj = GridFIASettings(
            data_dir=tmp_path / "data",
            cache_dir=cache_dir
        )

        # temp_dir is None by default
        assert settings_obj.processing.temp_dir is None

        result = settings_obj.get_temp_path("temp_file.tmp")
        assert result == cache_dir / "temp_file.tmp"

    def test_path_string_conversion(self, tmp_path: Path, monkeypatch):
        """Test that string paths are converted to Path objects."""
        monkeypatch.chdir(tmp_path)

        settings_obj = GridFIASettings(
            data_dir=str(tmp_path / "data"),
            output_dir=str(tmp_path / "output"),
            cache_dir=str(tmp_path / "cache")
        )

        assert isinstance(settings_obj.data_dir, Path)
        assert isinstance(settings_obj.output_dir, Path)
        assert isinstance(settings_obj.cache_dir, Path)


class TestBigMapSettingsAlias:
    """Test BigMapSettings backwards compatibility alias."""

    def test_bigmap_settings_is_gridfia_settings(self):
        """Test that BigMapSettings is an alias for GridFIASettings."""
        assert BigMapSettings is GridFIASettings

    def test_bigmap_settings_creates_same_instance_type(self, tmp_path: Path, monkeypatch):
        """Test that BigMapSettings creates GridFIASettings instance."""
        monkeypatch.chdir(tmp_path)

        settings_obj = BigMapSettings(data_dir=tmp_path / "data")

        assert isinstance(settings_obj, GridFIASettings)
        assert isinstance(settings_obj, BigMapSettings)


class TestGlobalSettingsInstance:
    """Test the global settings instance."""

    def test_global_settings_exists(self):
        """Test that global settings instance exists."""
        assert settings is not None
        assert isinstance(settings, GridFIASettings)


# =============================================================================
# load_settings Tests
# =============================================================================


class TestLoadSettings:
    """Test load_settings function."""

    def test_load_from_json_file(self, tmp_path: Path, monkeypatch):
        """Test loading settings from JSON file."""
        monkeypatch.chdir(tmp_path)

        config_data = {
            "data_dir": str(tmp_path / "json_data"),
            "output_dir": str(tmp_path / "json_output"),
            "debug": True,
            "verbose": True,
            "visualization": {
                "default_dpi": 150,
                "font_size": 14
            }
        }

        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        loaded_settings = load_settings(config_file)

        assert loaded_settings.data_dir == Path(tmp_path / "json_data")
        assert loaded_settings.output_dir == Path(tmp_path / "json_output")
        assert loaded_settings.debug is True
        assert loaded_settings.verbose is True
        assert loaded_settings.visualization.default_dpi == 150
        assert loaded_settings.visualization.font_size == 14

    def test_load_from_yaml_file(self, tmp_path: Path, monkeypatch):
        """Test loading settings from YAML file."""
        monkeypatch.chdir(tmp_path)

        config_content = f"""
data_dir: {tmp_path / "yaml_data"}
output_dir: {tmp_path / "yaml_output"}
debug: true
verbose: false
visualization:
  default_dpi: 200
  font_size: 10
calculations:
  - name: species_richness
    enabled: true
  - name: total_biomass
    enabled: false
"""

        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        loaded_settings = load_settings(config_file)

        assert loaded_settings.data_dir == Path(tmp_path / "yaml_data")
        assert loaded_settings.debug is True
        assert loaded_settings.verbose is False
        assert loaded_settings.visualization.default_dpi == 200
        assert len(loaded_settings.calculations) == 2
        assert loaded_settings.calculations[0].name == "species_richness"
        assert loaded_settings.calculations[1].enabled is False

    def test_load_from_yml_extension(self, tmp_path: Path, monkeypatch):
        """Test loading settings from .yml file (alternative extension)."""
        monkeypatch.chdir(tmp_path)

        config_content = f"""
data_dir: {tmp_path / "yml_data"}
debug: true
"""

        config_file = tmp_path / "config.yml"
        config_file.write_text(config_content)

        loaded_settings = load_settings(config_file)

        assert loaded_settings.data_dir == Path(tmp_path / "yml_data")
        assert loaded_settings.debug is True

    def test_load_with_missing_file_returns_defaults(self, tmp_path: Path, monkeypatch):
        """Test that missing file returns default settings."""
        monkeypatch.chdir(tmp_path)

        missing_file = tmp_path / "nonexistent.json"
        loaded_settings = load_settings(missing_file)

        # Should return default settings
        assert isinstance(loaded_settings, GridFIASettings)
        assert loaded_settings.debug is False

    def test_load_with_none_returns_defaults(self, tmp_path: Path, monkeypatch):
        """Test that None config_file returns default settings."""
        monkeypatch.chdir(tmp_path)

        loaded_settings = load_settings(None)

        assert isinstance(loaded_settings, GridFIASettings)
        assert loaded_settings.app_name == "GridFIA"

    def test_load_with_invalid_json_raises_error(self, tmp_path: Path):
        """Test that invalid JSON raises error."""
        config_file = tmp_path / "invalid.json"
        config_file.write_text("{ invalid json content }")

        with pytest.raises(json.JSONDecodeError):
            load_settings(config_file)

    def test_load_with_invalid_yaml_raises_error(self, tmp_path: Path):
        """Test that invalid YAML raises error."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: content: [")

        with pytest.raises(yaml.YAMLError):
            load_settings(config_file)

    def test_load_with_invalid_field_values(self, tmp_path: Path, monkeypatch):
        """Test that invalid field values raise validation error."""
        monkeypatch.chdir(tmp_path)

        config_data = {
            "data_dir": str(tmp_path / "data"),
            "visualization": {
                "default_dpi": 9999  # Invalid: max is 600
            }
        }

        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        with pytest.raises(ValueError, match="less than or equal to 600"):
            load_settings(config_file)

    def test_load_with_nested_processing_config(self, tmp_path: Path, monkeypatch):
        """Test loading nested processing config."""
        monkeypatch.chdir(tmp_path)

        config_data = {
            "data_dir": str(tmp_path / "data"),
            "processing": {
                "max_workers": 8,
                "memory_limit_gb": 32.0
            }
        }

        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        loaded_settings = load_settings(config_file)

        assert loaded_settings.processing.max_workers == 8
        assert loaded_settings.processing.memory_limit_gb == 32.0

    def test_load_with_path_object(self, tmp_path: Path, monkeypatch):
        """Test loading with Path object instead of string."""
        monkeypatch.chdir(tmp_path)

        config_data = {"debug": True}
        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        # Pass as Path object
        loaded_settings = load_settings(Path(config_file))
        assert loaded_settings.debug is True


# =============================================================================
# save_settings Tests
# =============================================================================


class TestSaveSettings:
    """Test save_settings function."""

    def test_save_to_json_file(self, tmp_path: Path, monkeypatch):
        """Test saving settings to JSON file."""
        monkeypatch.chdir(tmp_path)

        settings_obj = GridFIASettings(
            data_dir=tmp_path / "data",
            output_dir=tmp_path / "output",
            debug=True,
            verbose=True
        )

        output_file = tmp_path / "saved_config.json"
        save_settings(settings_obj, output_file)

        assert output_file.exists()

        # Verify content
        with open(output_file) as f:
            saved_data = json.load(f)

        assert saved_data["debug"] is True
        assert saved_data["verbose"] is True
        assert saved_data["app_name"] == "GridFIA"

    def test_save_creates_parent_directories(self, tmp_path: Path, monkeypatch):
        """Test that save_settings creates parent directories."""
        monkeypatch.chdir(tmp_path)

        settings_obj = GridFIASettings(data_dir=tmp_path / "data")

        nested_output = tmp_path / "deep" / "nested" / "config.json"
        assert not nested_output.parent.exists()

        save_settings(settings_obj, nested_output)

        assert nested_output.exists()
        assert nested_output.parent.exists()

    def test_save_path_serialization(self, tmp_path: Path, monkeypatch):
        """Test that Path objects are serialized as strings."""
        monkeypatch.chdir(tmp_path)

        settings_obj = GridFIASettings(
            data_dir=tmp_path / "data",
            output_dir=tmp_path / "output",
            cache_dir=tmp_path / "cache"
        )

        output_file = tmp_path / "config.json"
        save_settings(settings_obj, output_file)

        with open(output_file) as f:
            saved_data = json.load(f)

        # Paths should be serialized as strings
        assert isinstance(saved_data["data_dir"], str)
        assert isinstance(saved_data["output_dir"], str)
        assert isinstance(saved_data["cache_dir"], str)

    def test_save_nested_configs(self, tmp_path: Path, monkeypatch):
        """Test that nested configs are properly saved."""
        monkeypatch.chdir(tmp_path)

        settings_obj = GridFIASettings(
            data_dir=tmp_path / "data",
            visualization=VisualizationConfig(default_dpi=150, font_size=16),
            processing=ProcessingConfig(max_workers=4, memory_limit_gb=16.0)
        )

        output_file = tmp_path / "config.json"
        save_settings(settings_obj, output_file)

        with open(output_file) as f:
            saved_data = json.load(f)

        assert saved_data["visualization"]["default_dpi"] == 150
        assert saved_data["visualization"]["font_size"] == 16
        assert saved_data["processing"]["max_workers"] == 4
        assert saved_data["processing"]["memory_limit_gb"] == 16.0

    def test_save_calculations_list(self, tmp_path: Path, monkeypatch):
        """Test that calculations list is properly saved."""
        monkeypatch.chdir(tmp_path)

        calcs = [
            CalculationConfig(name="calc1", enabled=True, parameters={"key": "value"}),
            CalculationConfig(name="calc2", enabled=False)
        ]

        settings_obj = GridFIASettings(
            data_dir=tmp_path / "data",
            calculations=calcs
        )

        output_file = tmp_path / "config.json"
        save_settings(settings_obj, output_file)

        with open(output_file) as f:
            saved_data = json.load(f)

        assert len(saved_data["calculations"]) == 2
        assert saved_data["calculations"][0]["name"] == "calc1"
        assert saved_data["calculations"][0]["enabled"] is True
        assert saved_data["calculations"][0]["parameters"] == {"key": "value"}
        assert saved_data["calculations"][1]["name"] == "calc2"
        assert saved_data["calculations"][1]["enabled"] is False

    def test_save_overwrites_existing_file(self, tmp_path: Path, monkeypatch):
        """Test that save_settings overwrites existing file."""
        monkeypatch.chdir(tmp_path)

        output_file = tmp_path / "config.json"

        # Create initial config
        settings1 = GridFIASettings(data_dir=tmp_path / "data", debug=False)
        save_settings(settings1, output_file)

        # Overwrite with new config
        settings2 = GridFIASettings(data_dir=tmp_path / "data", debug=True)
        save_settings(settings2, output_file)

        with open(output_file) as f:
            saved_data = json.load(f)

        assert saved_data["debug"] is True

    def test_save_and_load_roundtrip(self, tmp_path: Path, monkeypatch):
        """Test that settings can be saved and loaded back."""
        monkeypatch.chdir(tmp_path)

        original_settings = GridFIASettings(
            data_dir=tmp_path / "data",
            output_dir=tmp_path / "output",
            debug=True,
            verbose=True,
            visualization=VisualizationConfig(default_dpi=200, font_size=14),
            calculations=[
                CalculationConfig(name="test_calc", enabled=True, parameters={"x": 1})
            ]
        )

        config_file = tmp_path / "roundtrip.json"
        save_settings(original_settings, config_file)
        loaded_settings = load_settings(config_file)

        assert loaded_settings.debug == original_settings.debug
        assert loaded_settings.verbose == original_settings.verbose
        assert loaded_settings.visualization.default_dpi == original_settings.visualization.default_dpi
        assert loaded_settings.visualization.font_size == original_settings.visualization.font_size
        assert len(loaded_settings.calculations) == len(original_settings.calculations)
        assert loaded_settings.calculations[0].name == original_settings.calculations[0].name


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestEdgeCasesAndErrors:
    """Test edge cases and error handling."""

    def test_visualization_config_with_dict_input(self, tmp_path: Path, monkeypatch):
        """Test VisualizationConfig creation from dict."""
        monkeypatch.chdir(tmp_path)

        viz_dict = {
            "default_dpi": 150,
            "font_size": 14,
            "default_figure_size": [10, 8],
            "color_maps": {"custom": "magma"}
        }

        # Should work when passed to GridFIASettings
        settings_obj = GridFIASettings(
            data_dir=tmp_path / "data",
            visualization=viz_dict
        )

        assert settings_obj.visualization.default_dpi == 150
        assert settings_obj.visualization.font_size == 14

    def test_processing_config_with_dict_input(self, tmp_path: Path, monkeypatch):
        """Test ProcessingConfig creation from dict."""
        monkeypatch.chdir(tmp_path)

        proc_dict = {
            "max_workers": 4,
            "memory_limit_gb": 16.0
        }

        settings_obj = GridFIASettings(
            data_dir=tmp_path / "data",
            processing=proc_dict
        )

        assert settings_obj.processing.max_workers == 4
        assert settings_obj.processing.memory_limit_gb == 16.0

    def test_calculations_from_list_of_dicts(self, tmp_path: Path, monkeypatch):
        """Test calculations list from list of dicts."""
        monkeypatch.chdir(tmp_path)

        calcs_dicts = [
            {"name": "calc1", "enabled": True},
            {"name": "calc2", "enabled": False, "parameters": {"key": "value"}}
        ]

        settings_obj = GridFIASettings(
            data_dir=tmp_path / "data",
            calculations=calcs_dicts
        )

        assert len(settings_obj.calculations) == 2
        assert settings_obj.calculations[0].name == "calc1"
        assert settings_obj.calculations[1].parameters == {"key": "value"}

    def test_empty_calculations_list_raises_error(self, tmp_path: Path, monkeypatch):
        """Test that empty calculations list raises validation error (min_length=1)."""
        monkeypatch.chdir(tmp_path)

        with pytest.raises(ValueError, match="at least 1 item"):
            GridFIASettings(
                data_dir=tmp_path / "data",
                calculations=[]
            )

    def test_special_characters_in_paths(self, tmp_path: Path, monkeypatch):
        """Test paths with special characters."""
        monkeypatch.chdir(tmp_path)

        # Create path with spaces and special chars
        special_dir = tmp_path / "my data dir" / "with spaces"

        settings_obj = GridFIASettings(
            data_dir=special_dir
        )

        assert settings_obj.data_dir == special_dir
        assert special_dir.exists()

    def test_unicode_in_calculation_names(self, tmp_path: Path, monkeypatch):
        """Test unicode characters in calculation names."""
        monkeypatch.chdir(tmp_path)

        calcs = [
            CalculationConfig(name="diversite_shannon"),
            CalculationConfig(name="richesse_especes"),
            CalculationConfig(name="biomasse_totale")
        ]

        settings_obj = GridFIASettings(
            data_dir=tmp_path / "data",
            calculations=calcs
        )

        assert len(settings_obj.calculations) == 3

    def test_float_memory_limit_edge_cases(self):
        """Test memory limit float edge cases."""
        # Very small but valid
        config = ProcessingConfig(memory_limit_gb=0.001)
        assert config.memory_limit_gb == 0.001

        # Very large
        config = ProcessingConfig(memory_limit_gb=1024.0)
        assert config.memory_limit_gb == 1024.0

    def test_boundary_dpi_values(self):
        """Test exact boundary DPI values."""
        # Exactly at minimum
        config = VisualizationConfig(default_dpi=72)
        assert config.default_dpi == 72

        # Exactly at maximum
        config = VisualizationConfig(default_dpi=600)
        assert config.default_dpi == 600

    def test_boundary_font_size_values(self):
        """Test exact boundary font size values."""
        # Exactly at minimum
        config = VisualizationConfig(font_size=8)
        assert config.font_size == 8

        # Exactly at maximum
        config = VisualizationConfig(font_size=24)
        assert config.font_size == 24


class TestModelDump:
    """Test model_dump functionality for serialization."""

    def test_visualization_config_model_dump(self):
        """Test VisualizationConfig model_dump."""
        config = VisualizationConfig(default_dpi=150)
        dump = config.model_dump()

        assert isinstance(dump, dict)
        assert dump["default_dpi"] == 150
        assert "default_figure_size" in dump
        assert "color_maps" in dump
        assert "font_size" in dump

    def test_processing_config_model_dump(self, tmp_path: Path):
        """Test ProcessingConfig model_dump."""
        config = ProcessingConfig(max_workers=4, temp_dir=tmp_path / "temp")
        dump = config.model_dump()

        assert isinstance(dump, dict)
        assert dump["max_workers"] == 4
        assert dump["memory_limit_gb"] == 8.0
        # temp_dir should be Path object in dump
        assert dump["temp_dir"] == tmp_path / "temp"

    def test_calculation_config_model_dump(self):
        """Test CalculationConfig model_dump."""
        config = CalculationConfig(
            name="test",
            enabled=True,
            parameters={"key": "value"}
        )
        dump = config.model_dump()

        assert isinstance(dump, dict)
        assert dump["name"] == "test"
        assert dump["enabled"] is True
        assert dump["parameters"] == {"key": "value"}
        # OutputFormat enum is serialized to its string value
        assert dump["output_format"] == OutputFormat.GEOTIFF

    def test_gridfia_settings_model_dump(self, tmp_path: Path, monkeypatch):
        """Test GridFIASettings model_dump."""
        monkeypatch.chdir(tmp_path)

        settings_obj = GridFIASettings(
            data_dir=tmp_path / "data",
            debug=True,
            calculations=[
                CalculationConfig(name="calc1"),
                CalculationConfig(name="calc2")
            ]
        )
        dump = settings_obj.model_dump()

        assert isinstance(dump, dict)
        assert dump["debug"] is True
        assert dump["app_name"] == "GridFIA"
        assert isinstance(dump["visualization"], dict)
        assert isinstance(dump["processing"], dict)
        assert isinstance(dump["calculations"], list)
        assert len(dump["calculations"]) == 2
