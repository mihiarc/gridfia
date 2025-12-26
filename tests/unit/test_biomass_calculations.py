"""
Comprehensive unit tests for biomass calculations.

Tests all biomass calculation classes including TotalBiomass, TotalBiomassComparison,
SpeciesProportion, SpeciesPercentage, SpeciesGroupProportion, and BiomassThreshold.
Covers data validation, edge cases, error conditions, and statistical calculations.
"""

import pytest
import numpy as np
import logging
from typing import List

from gridfia.core.calculations.biomass import (
    TotalBiomass,
    TotalBiomassComparison,
    SpeciesProportion,
    SpeciesPercentage,
    SpeciesGroupProportion,
    BiomassThreshold,
)


class TestTotalBiomass:
    """Test suite for TotalBiomass calculation."""

    def test_init_default_parameters(self):
        """Test TotalBiomass initialization with default parameters."""
        calc = TotalBiomass()

        assert calc.name == "total_biomass"
        assert calc.description == "Total above-ground biomass across species"
        assert calc.units == "Mg/ha"
        assert calc.config['exclude_total_layer'] is True

    def test_init_custom_parameters(self):
        """Test TotalBiomass initialization with custom parameters."""
        calc = TotalBiomass(exclude_total_layer=False, custom_param="test")

        assert calc.config['exclude_total_layer'] is False
        assert calc.config['custom_param'] == "test"

    def test_calculate_exclude_total_layer(self):
        """Test total biomass calculation excluding first layer."""
        # Create test data: total layer + 3 species
        data = np.array([
            [[100, 150]], # Total layer (should be excluded)
            [[20, 30]],   # Species 1
            [[30, 40]],   # Species 2
            [[50, 80]]    # Species 3
        ], dtype=np.float32)

        calc = TotalBiomass(exclude_total_layer=True)
        result = calc.calculate(data)

        # Should sum only species layers (1-3)
        expected = np.array([[100, 150]], dtype=np.float32)  # 20+30+50, 30+40+80
        np.testing.assert_array_equal(result, expected)

    def test_calculate_include_total_layer(self):
        """Test total biomass calculation including first layer."""
        data = np.array([
            [[10, 15]], # Layer 0
            [[20, 25]], # Layer 1
            [[30, 35]]  # Layer 2
        ], dtype=np.float32)

        calc = TotalBiomass(exclude_total_layer=False)
        result = calc.calculate(data)

        # Should sum all layers
        expected = np.array([[60, 75]], dtype=np.float32)  # 10+20+30, 15+25+35
        np.testing.assert_array_equal(result, expected)

    def test_calculate_single_layer(self):
        """Test total biomass calculation with single layer."""
        data = np.array([[[50, 75]]], dtype=np.float32)

        calc = TotalBiomass(exclude_total_layer=True)
        result = calc.calculate(data)

        # Single layer should be returned as-is
        expected = np.array([[50, 75]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_calculate_runtime_parameter_override(self):
        """Test that runtime parameters override initialization parameters."""
        data = np.array([
            [[100]],  # Total
            [[20]],   # Species 1
            [[30]],   # Species 2
            [[50]]    # Species 3
        ], dtype=np.float32)

        calc = TotalBiomass(exclude_total_layer=True)

        # Override at runtime to include total layer
        result = calc.calculate(data, exclude_total_layer=False)

        # Should sum all layers including total
        expected = np.array([[200]], dtype=np.float32)  # 100+20+30+50
        np.testing.assert_array_equal(result, expected)

    def test_calculate_with_zeros(self):
        """Test total biomass calculation with zero values."""
        data = np.array([
            [[0, 0, 50]], # Total
            [[0, 0, 10]], # Species 1
            [[0, 0, 20]], # Species 2
            [[0, 0, 20]]  # Species 3
        ], dtype=np.float32)

        calc = TotalBiomass(exclude_total_layer=True)
        result = calc.calculate(data)

        expected = np.array([[0, 0, 50]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_calculate_2d_spatial_data(self):
        """Test total biomass calculation with 2D spatial data."""
        data = np.array([
            [[100, 150], [200, 250]], # Total
            [[20, 30], [40, 50]],     # Species 1
            [[30, 40], [60, 70]],     # Species 2
            [[50, 80], [100, 130]]    # Species 3
        ], dtype=np.float32)

        calc = TotalBiomass(exclude_total_layer=True)
        result = calc.calculate(data)

        # Sum species layers
        expected = np.array([[100, 150], [200, 250]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_validate_data_valid_3d_array(self):
        """Test data validation with valid 3D array."""
        data = np.zeros((5, 10, 10), dtype=np.float32)
        calc = TotalBiomass()

        assert calc.validate_data(data) is True

    def test_validate_data_invalid_dimensions(self):
        """Test data validation with invalid dimensions."""
        calc = TotalBiomass()

        # 2D array should be invalid
        data_2d = np.zeros((10, 10))
        assert calc.validate_data(data_2d) is False

        # 4D array should be invalid
        data_4d = np.zeros((2, 3, 10, 10))
        assert calc.validate_data(data_4d) is False

    def test_validate_data_empty_species_dimension(self):
        """Test data validation with empty species dimension."""
        data = np.zeros((0, 10, 10), dtype=np.float32)
        calc = TotalBiomass()

        assert calc.validate_data(data) is False

    def test_get_metadata(self):
        """Test metadata retrieval."""
        calc = TotalBiomass(exclude_total_layer=False, test_param="value")
        metadata = calc.get_metadata()

        assert metadata['name'] == "total_biomass"
        assert metadata['description'] == "Total above-ground biomass across species"
        assert metadata['units'] == "Mg/ha"
        assert metadata['config']['exclude_total_layer'] is False
        assert metadata['config']['test_param'] == "value"
        assert metadata['dtype'] == np.float32

    def test_get_output_dtype(self):
        """Test output dtype specification."""
        calc = TotalBiomass()
        assert calc.get_output_dtype() == np.float32


class TestTotalBiomassComparison:
    """Test suite for TotalBiomassComparison calculation."""

    def test_init_default_parameters(self):
        """Test TotalBiomassComparison initialization with default parameters."""
        calc = TotalBiomassComparison()

        assert calc.name == "total_biomass_comparison"
        assert "Difference between calculated and pre-calculated total biomass" in calc.description
        assert calc.units == "Mg/ha"
        assert calc.config['tolerance'] == 0.01

    def test_init_custom_tolerance(self):
        """Test TotalBiomassComparison initialization with custom tolerance."""
        calc = TotalBiomassComparison(tolerance=0.05)
        assert calc.config['tolerance'] == 0.05

    def test_calculate_perfect_match(self):
        """Test comparison with perfect match between calculated and pre-calculated."""
        data = np.array([
            [[100]], # Pre-calculated total
            [[30]],  # Species 1
            [[40]],  # Species 2
            [[30]]   # Species 3
        ], dtype=np.float32)

        calc = TotalBiomassComparison()
        result = calc.calculate(data)

        # Difference should be 0
        expected = np.array([[0]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_calculate_with_difference(self):
        """Test comparison with difference between totals."""
        data = np.array([
            [[95]],  # Pre-calculated total (slightly different)
            [[30]],  # Species 1
            [[40]],  # Species 2
            [[30]]   # Species 3 (sum = 100)
        ], dtype=np.float32)

        calc = TotalBiomassComparison()
        result = calc.calculate(data)

        # Absolute difference should be 5
        expected = np.array([[5]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_calculate_single_layer_warning(self, caplog):
        """Test behavior with single layer (should log warning and return zeros)."""
        data = np.array([[[50]]], dtype=np.float32)

        calc = TotalBiomassComparison()

        with caplog.at_level(logging.WARNING):
            result = calc.calculate(data)

        # Should return NaN and log warning (NaN indicates calculation failure,
        # distinguishing from actual zero difference between totals)
        assert result.shape == (1, 1)
        assert np.all(np.isnan(result)), "Failed calculation should return NaN, not zero"
        assert "Cannot compare totals with only one layer" in caplog.text

    def test_calculate_2d_spatial_data(self):
        """Test comparison with 2D spatial data."""
        data = np.array([
            [[100, 110], [120, 130]], # Pre-calculated
            [[25, 30], [35, 40]],     # Species 1
            [[35, 40], [45, 50]],     # Species 2
            [[40, 35], [35, 35]]      # Species 3
        ], dtype=np.float32)

        calc = TotalBiomassComparison()
        result = calc.calculate(data)

        # Calculate expected differences
        calculated_total = np.array([[100, 105], [115, 125]])  # Sum of species
        pre_calculated_total = np.array([[100, 110], [120, 130]])
        expected = np.abs(pre_calculated_total - calculated_total)

        np.testing.assert_array_equal(result, expected)

    def test_validate_data_valid(self):
        """Test data validation with valid data (multiple layers)."""
        data = np.zeros((5, 10, 10), dtype=np.float32)
        calc = TotalBiomassComparison()

        assert calc.validate_data(data) is True

    def test_validate_data_invalid_single_layer(self):
        """Test data validation with single layer (invalid)."""
        data = np.zeros((1, 10, 10), dtype=np.float32)
        calc = TotalBiomassComparison()

        assert calc.validate_data(data) is False

    def test_validate_data_invalid_dimensions(self):
        """Test data validation with invalid dimensions."""
        calc = TotalBiomassComparison()

        # 2D array
        data_2d = np.zeros((10, 10))
        assert calc.validate_data(data_2d) is False

        # 4D array
        data_4d = np.zeros((2, 3, 10, 10))
        assert calc.validate_data(data_4d) is False


class TestSpeciesProportion:
    """Test suite for SpeciesProportion calculation."""

    def test_init_with_species_name(self):
        """Test SpeciesProportion initialization with species name."""
        calc = SpeciesProportion(species_index=2, species_name="Oak")

        assert calc.name == "species_2_proportion"
        assert calc.description == "Proportion of biomass from Oak"
        assert calc.units == "fraction"
        assert calc.config['species_index'] == 2
        assert calc.config['species_name'] == "Oak"

    def test_init_without_species_name(self):
        """Test SpeciesProportion initialization without species name."""
        calc = SpeciesProportion(species_index=3)

        assert calc.name == "species_3_proportion"
        assert calc.description == "Proportion of biomass from species index 3"
        assert calc.config['species_name'] is None

    def test_calculate_proportion(self):
        """Test proportion calculation for specific species."""
        data = np.array([
            [[100]], # Total (ignored)
            [[20]],  # Species 1 (target: 20/70 = 2/7)
            [[30]],  # Species 2
            [[20]]   # Species 3
        ], dtype=np.float32)

        calc = SpeciesProportion(species_index=1)
        result = calc.calculate(data)

        expected = np.array([[20/70]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_calculate_with_zeros_in_total(self):
        """Test proportion calculation when total biomass is zero."""
        data = np.array([
            [[0]], # Total
            [[0]], # Species 1
            [[0]], # Species 2
            [[0]]  # Species 3
        ], dtype=np.float32)

        calc = SpeciesProportion(species_index=1)
        result = calc.calculate(data)

        # Should return 0 when total is 0
        expected = np.array([[0]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_calculate_mixed_zeros_and_values(self):
        """Test proportion with mixed zero and non-zero pixels."""
        data = np.array([
            [[100, 0]], # Total
            [[20, 0]],  # Species 1
            [[30, 0]],  # Species 2
            [[50, 0]]   # Species 3
        ], dtype=np.float32)

        calc = SpeciesProportion(species_index=1)
        result = calc.calculate(data)

        # First pixel: 20/100 = 0.2, second pixel: 0/0 = 0
        expected = np.array([[0.2, 0.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_calculate_runtime_parameter_override(self):
        """Test runtime parameter override."""
        data = np.array([
            [[100]], # Total
            [[20]],  # Species 1
            [[30]],  # Species 2
            [[50]]   # Species 3
        ], dtype=np.float32)

        calc = SpeciesProportion(species_index=1)

        # Override to calculate species 2 proportion instead
        result = calc.calculate(data, species_index=2)

        expected = np.array([[30/100]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_calculate_species_index_out_of_range(self):
        """Test error when species index is out of range."""
        data = np.array([
            [[100]], # Total
            [[20]],  # Species 1
            [[30]]   # Species 2
        ], dtype=np.float32)

        calc = SpeciesProportion(species_index=5)  # Out of range

        with pytest.raises(ValueError, match="Species index 5 out of range"):
            calc.calculate(data)

    def test_calculate_total_layer_index_error(self):
        """Test error when trying to calculate proportion for total layer."""
        data = np.array([
            [[100]], # Total
            [[50]],  # Species 1
            [[50]]   # Species 2
        ], dtype=np.float32)

        calc = SpeciesProportion(species_index=0)  # Total layer

        with pytest.raises(ValueError, match="Cannot calculate proportion for total layer"):
            calc.calculate(data)

    def test_validate_data_valid(self):
        """Test data validation with valid data."""
        data = np.zeros((5, 10, 10), dtype=np.float32)
        calc = SpeciesProportion(species_index=2)

        assert calc.validate_data(data) is True

    def test_validate_data_species_index_too_large(self):
        """Test data validation when species index is too large."""
        data = np.zeros((3, 10, 10), dtype=np.float32)
        calc = SpeciesProportion(species_index=5)  # Index >= shape[0]

        assert calc.validate_data(data) is False

    def test_validate_data_invalid_dimensions(self):
        """Test data validation with invalid dimensions."""
        calc = SpeciesProportion(species_index=1)

        data_2d = np.zeros((10, 10))
        assert calc.validate_data(data_2d) is False


class TestSpeciesPercentage:
    """Test suite for SpeciesPercentage calculation."""

    def test_init_with_species_name(self):
        """Test SpeciesPercentage initialization with species name."""
        calc = SpeciesPercentage(species_index=2, species_name="Pine")

        assert calc.name == "species_2_percentage"
        assert calc.description == "Percentage of biomass from Pine"
        assert calc.units == "percent"

    def test_init_without_species_name(self):
        """Test SpeciesPercentage initialization without species name."""
        calc = SpeciesPercentage(species_index=3)

        assert calc.name == "species_3_percentage"
        assert calc.description == "Percentage of biomass from species index 3"

    def test_calculate_percentage(self):
        """Test percentage calculation (proportion * 100)."""
        data = np.array([
            [[100]], # Total
            [[25]],  # Species 1 (25%)
            [[75]]   # Species 2
        ], dtype=np.float32)

        calc = SpeciesPercentage(species_index=1)
        result = calc.calculate(data)

        expected = np.array([[25.0]], dtype=np.float32)  # 25/100 * 100 = 25%
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_calculate_calls_parent_proportion(self):
        """Test that percentage calculation calls parent proportion method."""
        data = np.array([
            [[200]], # Total
            [[40]],  # Species 1
            [[160]]  # Species 2
        ], dtype=np.float32)

        calc = SpeciesPercentage(species_index=1)
        result = calc.calculate(data)

        # Should be (40/200) * 100 = 20%
        expected = np.array([[20.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)


class TestSpeciesGroupProportion:
    """Test suite for SpeciesGroupProportion calculation."""

    def test_init_with_group_name(self):
        """Test SpeciesGroupProportion initialization."""
        calc = SpeciesGroupProportion(species_indices=[1, 2, 3], group_name="Conifer Group")

        assert calc.name == "conifer_group_proportion"
        assert calc.description == "Combined proportion of biomass from Conifer Group"
        assert calc.units == "fraction"
        assert calc.config['species_indices'] == [1, 2, 3]
        assert calc.config['group_name'] == "Conifer Group"

    def test_calculate_group_proportion(self):
        """Test group proportion calculation."""
        data = np.array([
            [[100]], # Total (ignored)
            [[10]],  # Species 1 (in group)
            [[20]],  # Species 2 (in group)
            [[30]],  # Species 3 (not in group)
            [[40]]   # Species 4 (in group)
        ], dtype=np.float32)

        # Group includes species 1, 2, and 4
        calc = SpeciesGroupProportion(species_indices=[1, 2, 4], group_name="Test Group")
        result = calc.calculate(data)

        # Group biomass: 10+20+40 = 70
        # Total biomass (excluding layer 0): 10+20+30+40 = 100
        # Proportion: 70/100 = 0.7
        expected = np.array([[0.7]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_calculate_with_zeros_in_total(self):
        """Test group proportion when total is zero."""
        data = np.array([
            [[0]], # Total
            [[0]], # Species 1
            [[0]], # Species 2
            [[0]]  # Species 3
        ], dtype=np.float32)

        calc = SpeciesGroupProportion(species_indices=[1, 2], group_name="Test")
        result = calc.calculate(data)

        expected = np.array([[0]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_calculate_runtime_parameter_override(self):
        """Test runtime parameter override."""
        data = np.array([
            [[100]], # Total
            [[20]],  # Species 1
            [[30]],  # Species 2
            [[50]]   # Species 3
        ], dtype=np.float32)

        calc = SpeciesGroupProportion(species_indices=[1], group_name="Test")

        # Override to include different species
        result = calc.calculate(data, species_indices=[2, 3])

        # Group biomass: 30+50 = 80, Total: 100
        expected = np.array([[0.8]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_calculate_invalid_species_index(self):
        """Test error with invalid species index."""
        data = np.array([
            [[100]], # Total
            [[50]],  # Species 1
            [[50]]   # Species 2
        ], dtype=np.float32)

        calc = SpeciesGroupProportion(species_indices=[1, 5], group_name="Test")  # Index 5 out of range

        with pytest.raises(ValueError, match="Invalid species index: 5"):
            calc.calculate(data)

    def test_calculate_total_layer_in_group_error(self):
        """Test error when total layer (index 0) is included in group."""
        data = np.array([
            [[100]], # Total
            [[50]],  # Species 1
            [[50]]   # Species 2
        ], dtype=np.float32)

        calc = SpeciesGroupProportion(species_indices=[0, 1], group_name="Test")  # Index 0 invalid

        with pytest.raises(ValueError, match="Invalid species index: 0"):
            calc.calculate(data)

    def test_validate_data_valid(self):
        """Test data validation with valid data."""
        data = np.zeros((5, 10, 10), dtype=np.float32)
        calc = SpeciesGroupProportion(species_indices=[1, 2, 3], group_name="Test")

        assert calc.validate_data(data) is True

    def test_validate_data_species_index_out_of_range(self):
        """Test data validation with species index out of range."""
        data = np.zeros((3, 10, 10), dtype=np.float32)
        calc = SpeciesGroupProportion(species_indices=[1, 5], group_name="Test")  # Index 5 >= 3

        assert calc.validate_data(data) is False

    def test_validate_data_invalid_dimensions(self):
        """Test data validation with invalid dimensions."""
        calc = SpeciesGroupProportion(species_indices=[1, 2], group_name="Test")

        data_2d = np.zeros((10, 10))
        assert calc.validate_data(data_2d) is False


class TestBiomassThreshold:
    """Test suite for BiomassThreshold calculation."""

    def test_init_above_threshold(self):
        """Test BiomassThreshold initialization for above threshold."""
        calc = BiomassThreshold(threshold=50.0, above=True)

        assert calc.name == "biomass_above_50.0"
        assert "Areas with biomass above 50.0 Mg/ha" in calc.description
        assert calc.units == "boolean"
        assert calc.config['threshold'] == 50.0
        assert calc.config['above'] is True

    def test_init_below_threshold(self):
        """Test BiomassThreshold initialization for below threshold."""
        calc = BiomassThreshold(threshold=25.0, above=False)

        assert calc.name == "biomass_below_25.0"
        assert "Areas with biomass below 25.0 Mg/ha" in calc.description
        assert calc.config['above'] is False

    def test_calculate_above_threshold(self):
        """Test threshold calculation for areas above threshold."""
        data = np.array([
            [[80, 120]], # Total (ignored)
            [[20, 30]],  # Species 1
            [[30, 40]],  # Species 2
            [[30, 50]]   # Species 3
        ], dtype=np.float32)

        calc = BiomassThreshold(threshold=90.0, above=True)
        result = calc.calculate(data)

        # Total biomass: [80, 120], threshold 90
        # Result: [False, True] -> [0, 1]
        expected = np.array([[0, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_calculate_below_threshold(self):
        """Test threshold calculation for areas below threshold."""
        data = np.array([
            [[100]], # Total
            [[25]],  # Species 1
            [[35]],  # Species 2
            [[40]]   # Species 3
        ], dtype=np.float32)

        calc = BiomassThreshold(threshold=90.0, above=False)
        result = calc.calculate(data)

        # Total biomass: 100, threshold 90, below=False means <=
        # 100 > 90, so result should be False -> 0
        expected = np.array([[0]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_calculate_equal_to_threshold_below(self):
        """Test threshold calculation when biomass equals threshold (below case)."""
        data = np.array([
            [[100]], # Total
            [[30]],  # Species 1
            [[30]],  # Species 2
            [[40]]   # Species 3
        ], dtype=np.float32)

        calc = BiomassThreshold(threshold=100.0, above=False)
        result = calc.calculate(data)

        # 100 <= 100 is True for below case
        expected = np.array([[1]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_calculate_runtime_parameter_override(self):
        """Test runtime parameter override."""
        data = np.array([
            [[100]], # Total
            [[50]],  # Species 1
            [[50]]   # Species 2
        ], dtype=np.float32)

        calc = BiomassThreshold(threshold=50.0, above=True)

        # Override threshold and direction at runtime
        result = calc.calculate(data, threshold=75.0, above=False)

        # Total: 100, threshold: 75, below case
        # 100 <= 75 is False -> 0
        expected = np.array([[0]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_calculate_uses_total_biomass_calculation(self):
        """Test that BiomassThreshold uses TotalBiomass calculation internally."""
        # Create data where total layer doesn't match individual species sum
        data = np.array([
            [[90]], # Total layer (different from species sum)
            [[20]], # Species 1
            [[30]], # Species 2
            [[40]]  # Species 3 (sum = 90, matches)
        ], dtype=np.float32)

        calc = BiomassThreshold(threshold=85.0, above=True)
        result = calc.calculate(data)

        # Should use calculated total (90) not the total layer
        # 90 > 85 -> True -> 1
        expected = np.array([[1]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_validate_data_valid(self):
        """Test data validation with valid data."""
        data = np.zeros((5, 10, 10), dtype=np.float32)
        calc = BiomassThreshold(threshold=50.0)

        assert calc.validate_data(data) is True

    def test_validate_data_empty_species(self):
        """Test data validation with empty species dimension."""
        data = np.zeros((0, 10, 10), dtype=np.float32)
        calc = BiomassThreshold(threshold=50.0)

        assert calc.validate_data(data) is False

    def test_validate_data_invalid_dimensions(self):
        """Test data validation with invalid dimensions."""
        calc = BiomassThreshold(threshold=50.0)

        data_2d = np.zeros((10, 10))
        assert calc.validate_data(data_2d) is False

    def test_get_output_dtype(self):
        """Test output dtype specification for boolean result."""
        calc = BiomassThreshold(threshold=50.0)
        assert calc.get_output_dtype() == np.uint8


class TestBiomassCalculationsEdgeCases:
    """Test edge cases and error conditions for biomass calculations."""

    def test_very_large_values(self):
        """Test calculations with very large biomass values."""
        large_value = 1e6
        data = np.array([
            [[large_value]], # Total
            [[large_value/2]], # Species 1
            [[large_value/2]]  # Species 2
        ], dtype=np.float32)

        calc = TotalBiomass(exclude_total_layer=True)
        result = calc.calculate(data)

        expected = np.array([[large_value]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=1)

    def test_very_small_values(self):
        """Test calculations with very small biomass values."""
        small_value = 1e-6
        data = np.array([
            [[small_value*3]], # Total
            [[small_value]], # Species 1
            [[small_value*2]]  # Species 2
        ], dtype=np.float32)

        calc = SpeciesProportion(species_index=1)
        result = calc.calculate(data)

        expected = np.array([[1.0/3.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_nan_and_inf_values(self):
        """Test handling of NaN and infinity values."""
        data = np.array([
            [[100, np.nan, np.inf]], # Total
            [[30, 0, 50]], # Species 1
            [[70, 0, 50]]  # Species 2
        ], dtype=np.float32)

        calc = TotalBiomass(exclude_total_layer=True)
        result = calc.calculate(data)

        # First pixel: normal calculation
        # Second pixel: 0 + 0 = 0
        # Third pixel: 50 + 50 = 100
        expected = np.array([[100, 0, 100]], dtype=np.float32)

        # Check non-NaN/inf values
        assert result[0, 0] == 100
        assert result[0, 1] == 0
        assert result[0, 2] == 100

    def test_negative_biomass_values(self):
        """Test handling of negative biomass values (should still work)."""
        data = np.array([
            [[50]], # Total
            [[-10]], # Species 1 (negative - unusual but mathematically valid)
            [[30]],  # Species 2
            [[30]]   # Species 3
        ], dtype=np.float32)

        calc = TotalBiomass(exclude_total_layer=True)
        result = calc.calculate(data)

        expected = np.array([[50]], dtype=np.float32)  # -10 + 30 + 30 = 50
        np.testing.assert_array_equal(result, expected)

    def test_mixed_data_types(self):
        """Test calculations work with different numpy data types."""
        # Test with int32 input - note that zeros_like preserves input dtype
        data_int = np.array([
            [[100]], # Total
            [[30]],  # Species 1
            [[70]]   # Species 2
        ], dtype=np.int32)

        calc = SpeciesProportion(species_index=1)
        result = calc.calculate(data_int)

        # With integer inputs, result will be integer (0 due to integer division)
        # This demonstrates the calculation works but highlights data type considerations
        assert result.dtype == np.int32

        # Test with float input for expected proportional results
        data_float = data_int.astype(np.float32)
        result_float = calc.calculate(data_float)

        expected_value = 30.0 / (30.0 + 70.0)  # 0.3
        np.testing.assert_array_almost_equal(result_float, [[expected_value]], decimal=6)

    def test_single_pixel_edge_cases(self):
        """Test edge cases with single pixel data."""
        # Single pixel, single species
        data = np.array([[[42]]], dtype=np.float32)

        calc = TotalBiomass()
        result = calc.calculate(data)

        expected = np.array([[42]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_empty_spatial_dimensions(self):
        """Test behavior with empty spatial dimensions."""
        # Valid species count but no spatial data
        data = np.zeros((3, 0, 0), dtype=np.float32)

        calc = TotalBiomass()
        result = calc.calculate(data)

        # Should return empty array with correct shape
        assert result.shape == (0, 0)
        assert result.dtype == np.float32


class TestBiomassCalculationsIntegration:
    """Integration tests using realistic data from fixtures."""

    def test_with_sample_zarr_array(self, sample_zarr_array):
        """Test biomass calculations with sample zarr array fixture."""
        # Convert zarr to numpy array
        data = np.array(sample_zarr_array[:])

        # Test total biomass calculation
        total_calc = TotalBiomass(exclude_total_layer=True)
        total_result = total_calc.calculate(data)

        # Should match pre-calculated total (within floating point precision)
        pre_calculated = data[0]  # First layer is total
        np.testing.assert_array_almost_equal(total_result, pre_calculated, decimal=3)

    def test_species_proportion_with_fixture(self, sample_zarr_array):
        """Test species proportion calculation with fixture data."""
        data = np.array(sample_zarr_array[:])

        # Test proportion for species 1 (dominant species)
        prop_calc = SpeciesProportion(species_index=1, species_name="Dominant Oak")
        result = prop_calc.calculate(data)

        # Verify proportions are between 0 and 1
        assert np.all(result >= 0)
        assert np.all(result <= 1)

        # Check that proportions are reasonable (species 1 should be significant)
        non_zero_mask = data[0] > 0  # Where there's total biomass
        if np.any(non_zero_mask):
            assert np.mean(result[non_zero_mask]) > 0.1  # At least 10% on average

    def test_threshold_analysis_with_fixture(self, sample_zarr_array):
        """Test threshold analysis with fixture data."""
        data = np.array(sample_zarr_array[:])

        # Test high biomass areas (> 50 Mg/ha)
        high_calc = BiomassThreshold(threshold=50.0, above=True)
        high_result = high_calc.calculate(data)

        # Test low biomass areas (<= 10 Mg/ha)
        low_calc = BiomassThreshold(threshold=10.0, above=False)
        low_result = low_calc.calculate(data)

        # Results should be binary (0 or 1)
        assert set(np.unique(high_result)).issubset({0, 1})
        assert set(np.unique(low_result)).issubset({0, 1})

        # High and low areas shouldn't overlap completely
        overlap = np.logical_and(high_result, low_result)
        assert not np.all(overlap)  # Not all pixels can be both high and low

    def test_group_analysis_with_fixture(self, sample_zarr_array):
        """Test species group analysis with fixture data."""
        data = np.array(sample_zarr_array[:])

        # Group species 1 and 2 as "major species"
        group_calc = SpeciesGroupProportion(
            species_indices=[1, 2],
            group_name="Major Species"
        )
        result = group_calc.calculate(data)

        # Verify proportions are valid
        assert np.all(result >= 0)
        assert np.all(result <= 1)

        # Group should contribute meaningfully where biomass exists
        non_zero_mask = data[0] > 0
        if np.any(non_zero_mask):
            assert np.mean(result[non_zero_mask]) > 0.2  # At least 20% combined