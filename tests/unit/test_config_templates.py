"""Tests for configuration template generator.

Tests template generation for analysis, species, and data configurations.
"""

import pytest
from pathlib import Path

from gridfia.config_templates import (
    create_config_template,
    list_available_templates,
    generate_example_configs,
    _create_analysis_template,
    _create_species_template,
    _create_data_template,
)


class TestCreateConfigTemplate:
    """Tests for the main create_config_template function."""

    def test_analysis_template_type(self):
        """Test creating analysis template."""
        config = create_config_template("analysis")

        assert "name" in config
        assert "description" in config
        assert "calculations" in config

    def test_species_template_type(self):
        """Test creating species template."""
        config = create_config_template("species")

        assert "species" in config
        assert "codes" in config["species"]
        assert "names" in config["species"]

    def test_data_template_type(self):
        """Test creating data template."""
        config = create_config_template("data")

        assert "data_processing" in config
        assert "type" in config["data_processing"]

    def test_unknown_type_raises_error(self):
        """Test that unknown config type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown config type"):
            create_config_template("unknown_type")

    def test_saves_to_file(self, tmp_path):
        """Test saving config to file."""
        output_path = tmp_path / "test_config.yaml"

        config = create_config_template("analysis", output_path=output_path)

        assert output_path.exists()
        assert config is not None

    def test_creates_parent_dirs(self, tmp_path):
        """Test that parent directories are created."""
        output_path = tmp_path / "nested" / "path" / "config.yaml"

        create_config_template("analysis", output_path=output_path)

        assert output_path.exists()

    def test_kwargs_passed_through(self):
        """Test that kwargs are passed to template function."""
        config = create_config_template(
            "analysis",
            name="custom_name",
            calculations=["shannon_diversity"],
        )

        assert config["name"] == "custom_name"


class TestAnalysisTemplate:
    """Tests for analysis template generation."""

    def test_default_values(self):
        """Test default analysis template values."""
        config = _create_analysis_template()

        assert config["name"] == "custom_analysis"
        assert config["description"] == "Custom forest analysis configuration"
        assert "output_dir" in config
        assert isinstance(config["calculations"], list)

    def test_default_calculations(self):
        """Test default calculations are species_richness and total_biomass."""
        config = _create_analysis_template()

        calc_names = [c["name"] for c in config["calculations"]]
        assert "species_richness" in calc_names
        assert "total_biomass" in calc_names

    def test_custom_name(self):
        """Test custom analysis name."""
        config = _create_analysis_template(name="my_analysis")

        assert config["name"] == "my_analysis"
        assert "my_analysis" in config["output_dir"]

    def test_custom_description(self):
        """Test custom description."""
        config = _create_analysis_template(description="My custom analysis")

        assert config["description"] == "My custom analysis"

    def test_custom_calculations(self):
        """Test specifying custom calculations."""
        config = _create_analysis_template(
            calculations=["shannon_diversity", "simpson_diversity", "evenness"]
        )

        calc_names = [c["name"] for c in config["calculations"]]
        assert "shannon_diversity" in calc_names
        assert "simpson_diversity" in calc_names
        assert "evenness" in calc_names
        assert len(config["calculations"]) == 3

    def test_unknown_calculation_skipped(self):
        """Test that unknown calculations are silently skipped."""
        config = _create_analysis_template(
            calculations=["species_richness", "unknown_calc"]
        )

        calc_names = [c["name"] for c in config["calculations"]]
        assert "species_richness" in calc_names
        assert "unknown_calc" not in calc_names
        assert len(config["calculations"]) == 1

    def test_calculation_parameters_included(self):
        """Test that calculation parameters are included."""
        config = _create_analysis_template(calculations=["shannon_diversity"])

        shannon_calc = config["calculations"][0]
        assert shannon_calc["name"] == "shannon_diversity"
        assert shannon_calc["enabled"] is True
        assert "parameters" in shannon_calc
        assert shannon_calc["parameters"]["base"] == "e"

    def test_dominant_species_calculation(self):
        """Test dominant_species calculation template."""
        config = _create_analysis_template(calculations=["dominant_species"])

        calc = config["calculations"][0]
        assert calc["name"] == "dominant_species"
        assert calc["parameters"]["min_biomass"] == 0.0

    def test_additional_kwargs(self):
        """Test that additional kwargs are added to config."""
        config = _create_analysis_template(custom_param="custom_value")

        assert config["custom_param"] == "custom_value"


class TestSpeciesTemplate:
    """Tests for species template generation."""

    def test_default_values(self):
        """Test default species template values."""
        config = _create_species_template()

        assert config["species"]["codes"] == [131]  # Loblolly Pine
        assert config["species"]["names"] == ["Loblolly Pine"]
        assert config["species"]["group_name"] == "species_group"

    def test_custom_species_codes(self):
        """Test custom species codes."""
        config = _create_species_template(species_codes=[202, 316, 318])

        assert config["species"]["codes"] == [202, 316, 318]

    def test_custom_species_names(self):
        """Test custom species names."""
        names = ["Douglas Fir", "Ponderosa Pine"]
        config = _create_species_template(species_names=names)

        assert config["species"]["names"] == names

    def test_custom_group_name(self):
        """Test custom group name."""
        config = _create_species_template(group_name="western_conifers")

        assert config["species"]["group_name"] == "western_conifers"
        assert "western_conifers" in config["name"]
        assert "western_conifers" in config["output_dir"]

    def test_calculations_include_species_proportion(self):
        """Test that species_proportion calculation is included."""
        config = _create_species_template(species_codes=[131, 318])

        calc_names = [c["name"] for c in config["calculations"]]
        assert "species_proportion" in calc_names
        assert "species_group_proportion" in calc_names

    def test_species_indices_calculated_correctly(self):
        """Test that species indices are calculated correctly."""
        config = _create_species_template(species_codes=[131, 318, 111])

        proportion_calc = next(
            c for c in config["calculations"] if c["name"] == "species_proportion"
        )
        # Indices should be 1-based (1, 2, 3) for 3 species
        assert proportion_calc["parameters"]["species_indices"] == [1, 2, 3]

    def test_additional_kwargs(self):
        """Test that additional kwargs are added."""
        config = _create_species_template(extra_param="value")

        assert config["extra_param"] == "value"


class TestDataTemplate:
    """Tests for data processing template generation."""

    def test_default_values(self):
        """Test default data template values."""
        config = _create_data_template()

        assert config["data_processing"]["type"] == "raster"
        assert config["data_processing"]["input_pattern"] == "*.tif"
        assert config["data_processing"]["chunk_size"] == [1000, 1000]
        assert config["data_processing"]["compression"] == "lz4"
        assert config["data_processing"]["overwrite"] is False

    def test_raster_type_settings(self):
        """Test raster-specific settings."""
        config = _create_data_template(data_type="raster")

        assert config["data_processing"]["crs"] == "EPSG:102039"
        assert config["data_processing"]["pixel_size_meters"] == 30.0
        assert config["data_processing"]["nodata_value"] == -9999

    def test_vector_type_settings(self):
        """Test vector-specific settings."""
        config = _create_data_template(data_type="vector")

        assert config["data_processing"]["type"] == "vector"
        assert config["data_processing"]["geometry_column"] == "geometry"
        assert config["data_processing"]["id_column"] == "parcel_id"

    def test_custom_input_pattern(self):
        """Test custom input pattern."""
        config = _create_data_template(input_pattern="nc_biomass_*.tif")

        assert config["data_processing"]["input_pattern"] == "nc_biomass_*.tif"

    def test_output_dir_includes_type(self):
        """Test output dir includes data type."""
        config = _create_data_template(data_type="raster")

        assert "raster" in config["output_dir"]

    def test_additional_kwargs(self):
        """Test that additional kwargs are added."""
        config = _create_data_template(custom_setting=True)

        assert config["custom_setting"] is True


class TestListAvailableTemplates:
    """Tests for list_available_templates function."""

    def test_returns_list(self):
        """Test that function returns a list."""
        templates = list_available_templates()

        assert isinstance(templates, list)

    def test_contains_expected_templates(self):
        """Test that all expected templates are listed."""
        templates = list_available_templates()

        assert "analysis" in templates
        assert "species" in templates
        assert "data" in templates

    def test_length(self):
        """Test correct number of templates."""
        templates = list_available_templates()

        assert len(templates) == 3


class TestGenerateExampleConfigs:
    """Tests for generate_example_configs function."""

    def test_creates_example_files(self, tmp_path):
        """Test that example files are created."""
        output_dir = tmp_path / "examples"

        generate_example_configs(output_dir=output_dir)

        assert output_dir.exists()
        # Check that files were created
        files = list(output_dir.glob("*.yaml"))
        assert len(files) == 3

    def test_creates_analysis_example(self, tmp_path):
        """Test that analysis example is created."""
        output_dir = tmp_path / "examples"

        generate_example_configs(output_dir=output_dir)

        assert (output_dir / "basic_analysis_config.yaml").exists()

    def test_creates_species_example(self, tmp_path):
        """Test that species example is created."""
        output_dir = tmp_path / "examples"

        generate_example_configs(output_dir=output_dir)

        assert (output_dir / "pine_species_config.yaml").exists()

    def test_creates_data_example(self, tmp_path):
        """Test that data example is created."""
        output_dir = tmp_path / "examples"

        generate_example_configs(output_dir=output_dir)

        assert (output_dir / "raster_processing_config.yaml").exists()

    def test_creates_parent_dirs(self, tmp_path):
        """Test that parent directories are created."""
        output_dir = tmp_path / "nested" / "path" / "examples"

        generate_example_configs(output_dir=output_dir)

        assert output_dir.exists()
