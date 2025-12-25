"""
Unit tests for species-specific calculations.

This module provides comprehensive test coverage for all species calculation classes
including DominantSpecies, SpeciesPresence, SpeciesDominance, RareSpecies, and CommonSpecies.
Tests cover normal operation, edge cases, error conditions, and parameter handling.
"""

import pytest
import numpy as np
from unittest.mock import patch
from gridfia.core.calculations.species import (
    DominantSpecies,
    SpeciesPresence,
    SpeciesDominance,
    RareSpecies,
    CommonSpecies
)


class TestDominantSpecies:
    """Test suite for DominantSpecies calculation."""

    def test_dominant_species_basic(self):
        """Test basic dominant species identification."""
        # Create test data: 4 species, 2x2 grid
        # Species 0: Total (should be excluded by default)
        # Species 1-3: Individual species
        data = np.array([
            [[60, 50], [40, 30]],  # Total layer
            [[10, 20], [30, 10]],  # Species 1
            [[30, 20], [5, 15]],   # Species 2 (dominant in most pixels)
            [[20, 10], [5, 5]]     # Species 3
        ], dtype=np.float32)

        calc = DominantSpecies()
        result = calc.calculate(data)

        # Check expected dominant species (1-indexed since total excluded)
        # Pixel (0,0): species 2 has 30 > 10, 20 -> index 2
        # Pixel (0,1): species 1 and 2 tied at 20 -> argmax gives index 1
        # Pixel (1,0): species 1 has 30 > 5, 5 -> index 1
        # Pixel (1,1): species 2 has 15 > 10, 5 -> index 2
        assert result[0, 0] == 2  # Species 2 dominant
        assert result[0, 1] == 1  # Species 1 dominant (tie goes to first)
        assert result[1, 0] == 1  # Species 1 dominant
        assert result[1, 1] == 2  # Species 2 dominant

    def test_dominant_species_with_min_biomass(self):
        """Test dominant species with minimum biomass threshold."""
        data = np.array([
            [[20, 5]],   # Total
            [[10, 2]],   # Species 1
            [[10, 3]]    # Species 2
        ], dtype=np.float32)

        calc = DominantSpecies(min_biomass=5.0)
        result = calc.calculate(data)

        # First pixel: both species above threshold, tie -> species 1
        # Second pixel: both below threshold -> 0
        assert result[0, 0] == 1
        assert result[0, 1] == 0

    def test_dominant_species_exclude_total_false(self):
        """Test dominant species including total layer."""
        data = np.array([
            [[100]],  # Total (should be dominant)
            [[10]],   # Species 1
            [[20]]    # Species 2
        ], dtype=np.float32)

        calc = DominantSpecies(exclude_total_layer=False)
        result = calc.calculate(data)

        # Total layer (index 0) should be dominant
        assert result[0, 0] == 0

    def test_dominant_species_single_layer(self):
        """Test dominant species with only one layer."""
        data = np.array([
            [[100, 50], [25, 0]]
        ], dtype=np.float32)

        calc = DominantSpecies(exclude_total_layer=True)
        result = calc.calculate(data)

        # With single layer and exclude_total=True, should still work
        # All non-zero pixels get index 0
        assert result[0, 0] == 0
        assert result[0, 1] == 0
        assert result[1, 0] == 0
        assert result[1, 1] == 0

    def test_dominant_species_all_zeros(self):
        """Test dominant species with all zero values."""
        data = np.zeros((3, 2, 2), dtype=np.float32)

        calc = DominantSpecies()
        result = calc.calculate(data)

        # All zeros should result in index 0 everywhere
        np.testing.assert_array_equal(result, np.zeros((2, 2), dtype=np.uint8))

    def test_dominant_species_dtype_output(self):
        """Test that dominant species returns uint8."""
        data = np.array([
            [[10]],
            [[20]]
        ], dtype=np.float32)

        calc = DominantSpecies()
        result = calc.calculate(data)

        assert result.dtype == np.uint8
        assert calc.get_output_dtype() == np.uint8

    def test_dominant_species_validation(self):
        """Test data validation for dominant species."""
        calc = DominantSpecies()

        # Valid 3D array
        valid_data = np.zeros((3, 10, 10))
        assert calc.validate_data(valid_data) is True

        # Invalid 2D array
        invalid_2d = np.zeros((10, 10))
        assert calc.validate_data(invalid_2d) is False

        # Empty array
        empty_data = np.zeros((0, 10, 10))
        assert calc.validate_data(empty_data) is False

    def test_dominant_species_kwargs_override(self):
        """Test that calculation kwargs override instance configuration."""
        data = np.array([
            [[50]],  # Total
            [[10]],  # Species 1
            [[20]]   # Species 2
        ], dtype=np.float32)

        calc = DominantSpecies(exclude_total_layer=True, min_biomass=0.0)

        # Override to include total layer
        result = calc.calculate(data, exclude_total_layer=False)
        assert result[0, 0] == 0  # Total layer dominant

        # Override minimum biomass
        result = calc.calculate(data, min_biomass=25.0)
        assert result[0, 0] == 0  # No species above threshold

    def test_dominant_species_metadata(self):
        """Test dominant species metadata."""
        calc = DominantSpecies(min_biomass=5.0)
        metadata = calc.get_metadata()

        assert metadata['name'] == 'dominant_species'
        assert metadata['description'] == 'Index of species with maximum biomass'
        assert metadata['units'] == 'species_index'
        assert metadata['config']['min_biomass'] == 5.0
        assert metadata['dtype'] == np.uint8


class TestSpeciesPresence:
    """Test suite for SpeciesPresence calculation."""

    def test_species_presence_basic(self):
        """Test basic species presence detection."""
        data = np.array([
            [[10, 0], [5, 15]],   # Species 0
            [[0, 20], [10, 0]],   # Species 1
            [[30, 5], [0, 25]]    # Species 2
        ], dtype=np.float32)

        calc = SpeciesPresence(species_index=1, threshold=0.0)
        result = calc.calculate(data)

        # Species 1 presence: only where biomass > 0
        expected = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_species_presence_with_threshold(self):
        """Test species presence with biomass threshold."""
        data = np.array([
            [[10, 2], [8, 15]]  # Single species
        ], dtype=np.float32)

        calc = SpeciesPresence(species_index=0, threshold=5.0)
        result = calc.calculate(data)

        # Only pixels with biomass > 5.0 are present
        expected = np.array([[1, 0], [1, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_species_presence_with_name(self):
        """Test species presence initialization with species name."""
        calc = SpeciesPresence(species_index=2, species_name="Douglas Fir", threshold=1.0)

        assert calc.name == "species_2_presence"
        assert "Douglas Fir" in calc.description
        assert calc.config['species_name'] == "Douglas Fir"
        assert calc.config['species_index'] == 2
        assert calc.config['threshold'] == 1.0

    def test_species_presence_without_name(self):
        """Test species presence initialization without species name."""
        calc = SpeciesPresence(species_index=3)

        assert calc.name == "species_3_presence"
        assert "species index 3" in calc.description
        assert calc.config['species_name'] is None

    def test_species_presence_index_out_of_range(self):
        """Test species presence with invalid species index."""
        data = np.array([
            [[10]],  # Only 1 species
            [[20]]   # Index 1
        ], dtype=np.float32)

        calc = SpeciesPresence(species_index=5)  # Index 5 doesn't exist

        with pytest.raises(ValueError, match="Species index 5 out of range"):
            calc.calculate(data)

    def test_species_presence_validation(self):
        """Test data validation for species presence."""
        calc = SpeciesPresence(species_index=2)

        # Valid data with enough species
        valid_data = np.zeros((5, 10, 10))
        assert calc.validate_data(valid_data) is True

        # Invalid - not enough species
        insufficient_data = np.zeros((2, 10, 10))
        assert calc.validate_data(insufficient_data) is False

        # Invalid 2D array
        invalid_2d = np.zeros((10, 10))
        assert calc.validate_data(invalid_2d) is False

    def test_species_presence_kwargs_override(self):
        """Test that kwargs override instance configuration."""
        data = np.array([
            [[10, 2]],
            [[5, 8]]
        ], dtype=np.float32)

        calc = SpeciesPresence(species_index=0, threshold=0.0)

        # Override species index
        result = calc.calculate(data, species_index=1)
        expected = np.array([[1, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

        # Override threshold
        result = calc.calculate(data, threshold=6.0)
        expected = np.array([[1, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_species_presence_output_dtype(self):
        """Test species presence output data type."""
        calc = SpeciesPresence(species_index=0)
        assert calc.get_output_dtype() == np.uint8

        data = np.array([[[10]]], dtype=np.float32)
        result = calc.calculate(data)
        assert result.dtype == np.uint8


class TestSpeciesDominance:
    """Test suite for SpeciesDominance calculation."""

    def test_species_dominance_basic(self):
        """Test basic species dominance calculation."""
        # 2x2 grid, species 1 dominant in 3/4 pixels
        data = np.array([
            [[60, 50], [40, 30]],  # Total
            [[30, 30], [30, 10]],  # Species 1 - dominant in 3 pixels
            [[20, 15], [5, 15]],   # Species 2 - dominant in 1 pixel
            [[10, 5], [5, 5]]      # Species 3 - never dominant
        ], dtype=np.float32)

        calc = SpeciesDominance(species_index=1)
        result = calc.calculate(data)

        # Species 1 dominant in 3/4 pixels = 75%
        expected_percentage = 75.0
        np.testing.assert_array_almost_equal(result,
                                           np.full((2, 2), expected_percentage, dtype=np.float32))

    def test_species_dominance_never_dominant(self):
        """Test species dominance when species is never dominant."""
        data = np.array([
            [[40, 40]],  # Total
            [[30, 30]],  # Species 1 - always dominant
            [[10, 10]]   # Species 2 - never dominant
        ], dtype=np.float32)

        calc = SpeciesDominance(species_index=2)
        result = calc.calculate(data)

        # Species 2 never dominant = 0%
        expected = np.zeros((1, 2), dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_species_dominance_always_dominant(self):
        """Test species dominance when species is always dominant."""
        data = np.array([
            [[50, 40]],  # Total
            [[40, 30]],  # Species 1 - always dominant
            [[10, 10]]   # Species 2
        ], dtype=np.float32)

        calc = SpeciesDominance(species_index=1)
        result = calc.calculate(data)

        # Species 1 always dominant = 100%
        expected = np.full((1, 2), 100.0, dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_species_dominance_with_name(self):
        """Test species dominance with species name."""
        calc = SpeciesDominance(species_index=3, species_name="White Pine")

        assert calc.name == "species_3_dominance"
        assert "White Pine" in calc.description
        assert calc.config['species_name'] == "White Pine"

    def test_species_dominance_invalid_index_zero(self):
        """Test species dominance with invalid index 0 (total layer)."""
        data = np.array([
            [[50]],  # Total
            [[30]],  # Species 1
            [[20]]   # Species 2
        ], dtype=np.float32)

        calc = SpeciesDominance(species_index=0)

        with pytest.raises(ValueError, match="Invalid species index: 0"):
            calc.calculate(data)

    def test_species_dominance_index_out_of_range(self):
        """Test species dominance with index out of range."""
        data = np.array([
            [[50]],  # Total
            [[30]],  # Species 1
            [[20]]   # Species 2
        ], dtype=np.float32)

        calc = SpeciesDominance(species_index=5)

        with pytest.raises(ValueError, match="Invalid species index: 5"):
            calc.calculate(data)

    def test_species_dominance_validation(self):
        """Test data validation for species dominance."""
        calc = SpeciesDominance(species_index=2)

        # Valid data
        valid_data = np.zeros((5, 10, 10))
        assert calc.validate_data(valid_data) is True

        # Invalid - not enough species
        insufficient_data = np.zeros((2, 10, 10))
        assert calc.validate_data(insufficient_data) is False

        # Invalid 2D array
        invalid_2d = np.zeros((10, 10))
        assert calc.validate_data(invalid_2d) is False

    def test_species_dominance_kwargs_override(self):
        """Test kwargs override for species dominance."""
        data = np.array([
            [[50, 40]],  # Total
            [[20, 25]],  # Species 1
            [[30, 15]]   # Species 2
        ], dtype=np.float32)

        calc = SpeciesDominance(species_index=1)

        # Override to check different species
        result = calc.calculate(data, species_index=2)

        # Species 2 dominant in 1/2 pixels = 50% (pixel 0: 30 > 20, pixel 1: 15 < 25)
        expected = np.full((1, 2), 50.0, dtype=np.float32)
        # Use almost_equal due to float precision
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_species_dominance_single_pixel(self):
        """Test species dominance with single pixel."""
        data = np.array([
            [[30]],  # Total
            [[20]],  # Species 1 - dominant
            [[10]]   # Species 2
        ], dtype=np.float32)

        calc = SpeciesDominance(species_index=1)
        result = calc.calculate(data)

        # Species 1 dominant in 1/1 pixels = 100%
        assert result[0, 0] == 100.0

    def test_species_dominance_uses_dominant_species_calc(self):
        """Test that species dominance uses DominantSpecies calculation internally."""
        data = np.array([
            [[50, 40]],  # Total
            [[30, 25]],  # Species 1
            [[20, 15]]   # Species 2
        ], dtype=np.float32)

        with patch('gridfia.core.calculations.species.DominantSpecies') as mock_dominant:
            mock_dominant.return_value.calculate.return_value = np.array([[1, 1]])

            calc = SpeciesDominance(species_index=1)
            result = calc.calculate(data)

            # Should have called DominantSpecies
            mock_dominant.assert_called_once()
            mock_dominant.return_value.calculate.assert_called_once_with(data)


class TestRareSpecies:
    """Test suite for RareSpecies calculation."""

    def test_rare_species_basic(self):
        """Test basic rare species identification."""
        # 3 species, 4 pixels total
        # Species pattern: widespread, patchy, rare
        data = np.array([
            [[40, 30, 20, 10]],  # Total
            [[20, 15, 10, 5]],   # Species 1: present in all 4 pixels (common)
            [[15, 10, 5, 0]],    # Species 2: present in 3/4 pixels (common)
            [[5, 5, 5, 5]]       # Species 3: present in 4/4 pixels (common)
        ], dtype=np.float32)

        # Set threshold so only species present in <25% of pixels are rare
        calc = RareSpecies(occurrence_threshold=0.25, biomass_threshold=0.0)
        result = calc.calculate(data)

        # No rare species with these patterns - all appear in ≥75% of pixels
        expected = np.zeros((1, 4), dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_rare_species_with_actual_rare(self):
        """Test rare species identification with truly rare species."""
        # 6 pixels, species with different occurrence patterns
        data = np.array([
            [[50, 40, 30, 20, 10, 0]],  # Total
            [[25, 20, 15, 10, 5, 0]],   # Species 1: 5/6 pixels (common)
            [[15, 10, 5, 0, 0, 0]],     # Species 2: 3/6 pixels (common)
            [[10, 0, 0, 0, 0, 0]],      # Species 3: 1/6 pixels (rare)
            [[0, 0, 10, 0, 0, 0]]       # Species 4: 1/6 pixels (rare)
        ], dtype=np.float32)

        # Species present in <20% of pixels are rare
        calc = RareSpecies(occurrence_threshold=0.2, biomass_threshold=0.0)
        result = calc.calculate(data)

        # Expected: species 3 and 4 are rare (each in 1/6 = 16.67% < 20%)
        # Pixel counts: [1, 0, 1, 0, 0, 0] (rare species present)
        expected = np.array([[1, 0, 1, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_rare_species_with_biomass_threshold(self):
        """Test rare species with biomass threshold."""
        data = np.array([
            [[20, 15, 10, 5]],    # Total
            [[15, 10, 5, 0]],     # Species 1: present in 3/4 pixels above threshold
            [[3, 3, 3, 3]],       # Species 2: present in all pixels but low biomass
            [[2, 2, 2, 2]]        # Species 3: below threshold everywhere
        ], dtype=np.float32)

        # Only biomass > 4.0 counts as "present"
        calc = RareSpecies(occurrence_threshold=0.5, biomass_threshold=4.0)
        result = calc.calculate(data)

        # Species 1: present in 3/4 pixels = 75% (not rare)
        # Species 2: present in 0/4 pixels above threshold = 0% (rare)
        # Species 3: present in 0/4 pixels above threshold = 0% (rare)
        # Both species 2 and 3 are rare, but not present above threshold anywhere
        expected = np.zeros((1, 4), dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_rare_species_exclude_total_layer(self):
        """Test that rare species calculation excludes total layer."""
        data = np.array([
            [[100, 100, 100, 100]],  # Total - would be common but should be excluded
            [[50, 0, 0, 0]],         # Species 1: 1/4 pixels (rare)
            [[25, 25, 25, 25]]       # Species 2: 4/4 pixels (common)
        ], dtype=np.float32)

        calc = RareSpecies(occurrence_threshold=0.3, biomass_threshold=0.0)
        result = calc.calculate(data)

        # Only species 1 is rare (25% occurrence)
        # Species 1 present only in first pixel
        expected = np.array([[1, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_rare_species_single_species_layer(self):
        """Test rare species with single species (no total layer)."""
        data = np.array([
            [[10, 0, 5, 0]]  # Single species
        ], dtype=np.float32)

        calc = RareSpecies(occurrence_threshold=0.6, biomass_threshold=0.0)
        result = calc.calculate(data)

        # Species present in 2/4 pixels = 50% < 60% threshold (rare)
        expected = np.array([[1, 0, 1, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_rare_species_all_common(self):
        """Test rare species when all species are common."""
        data = np.array([
            [[40, 40, 40, 40]],  # Total
            [[20, 20, 20, 20]],  # Species 1: everywhere
            [[20, 20, 20, 20]]   # Species 2: everywhere
        ], dtype=np.float32)

        calc = RareSpecies(occurrence_threshold=0.5, biomass_threshold=0.0)
        result = calc.calculate(data)

        # No rare species
        expected = np.zeros((1, 4), dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_rare_species_validation(self):
        """Test data validation for rare species."""
        calc = RareSpecies()

        # Valid 3D array
        valid_data = np.zeros((3, 10, 10))
        assert calc.validate_data(valid_data) is True

        # Invalid 2D array
        invalid_2d = np.zeros((10, 10))
        assert calc.validate_data(invalid_2d) is False

        # Empty array
        empty_data = np.zeros((0, 10, 10))
        assert calc.validate_data(empty_data) is False

    def test_rare_species_kwargs_override(self):
        """Test kwargs override for rare species parameters."""
        data = np.array([
            [[30, 20]],  # Total
            [[15, 0]],   # Species 1: 1/2 pixels = 50%
            [[15, 20]]   # Species 2: 2/2 pixels = 100%
        ], dtype=np.float32)

        calc = RareSpecies(occurrence_threshold=0.3, biomass_threshold=0.0)

        # Override occurrence threshold to make species 1 rare
        result = calc.calculate(data, occurrence_threshold=0.6)
        expected = np.array([[1, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

        # Override biomass threshold
        result = calc.calculate(data, biomass_threshold=16.0)
        # Only species 2 at pixel 2 is above threshold, but 1/2 = 50% > 30% so not rare
        expected = np.array([[0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_rare_species_output_dtype(self):
        """Test rare species output data type."""
        calc = RareSpecies()
        assert calc.get_output_dtype() == np.uint8

    def test_rare_species_metadata(self):
        """Test rare species metadata."""
        calc = RareSpecies(occurrence_threshold=0.05, biomass_threshold=2.0)
        metadata = calc.get_metadata()

        assert metadata['name'] == 'rare_species'
        assert metadata['description'] == 'Count of rare species per pixel'
        assert metadata['units'] == 'count'
        assert metadata['config']['occurrence_threshold'] == 0.05
        assert metadata['config']['biomass_threshold'] == 2.0


class TestCommonSpecies:
    """Test suite for CommonSpecies calculation."""

    def test_common_species_basic(self):
        """Test basic common species identification."""
        # 4 pixels, species with different occurrence patterns
        data = np.array([
            [[40, 30, 20, 10]],  # Total
            [[20, 15, 10, 5]],   # Species 1: 4/4 pixels = 100% (common)
            [[15, 10, 0, 0]],    # Species 2: 2/4 pixels = 50% (not common with 60% threshold)
            [[5, 0, 0, 0]]       # Species 3: 1/4 pixels = 25% (not common)
        ], dtype=np.float32)

        # Species in ≥60% of pixels are common
        calc = CommonSpecies(occurrence_threshold=0.6, biomass_threshold=0.0)
        result = calc.calculate(data)

        # Only species 1 is common (100% > 60%)
        # Present in all pixels
        expected = np.array([[1, 1, 1, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_common_species_multiple_common(self):
        """Test common species with multiple common species."""
        data = np.array([
            [[60, 50, 40, 30]],  # Total
            [[25, 20, 15, 10]],  # Species 1: 4/4 pixels (common)
            [[20, 15, 10, 0]],   # Species 2: 3/4 pixels = 75% (common)
            [[15, 15, 15, 20]]   # Species 3: 4/4 pixels (common)
        ], dtype=np.float32)

        # Species in ≥70% of pixels are common
        calc = CommonSpecies(occurrence_threshold=0.7, biomass_threshold=0.0)
        result = calc.calculate(data)

        # Species 1 and 3 are common (100%), species 2 is common (75%)
        # All three species at each pixel
        expected = np.array([[3, 3, 3, 2]], dtype=np.uint8)  # Pixel 4 missing species 2
        np.testing.assert_array_equal(result, expected)

    def test_common_species_with_biomass_threshold(self):
        """Test common species with biomass threshold."""
        data = np.array([
            [[30, 25, 20, 15]],  # Total
            [[15, 12, 10, 8]],   # Species 1: all above threshold, 4/4 pixels (common)
            [[8, 8, 5, 3]],      # Species 2: 2/4 pixels above threshold = 50%
            [[7, 5, 5, 4]]       # Species 3: 4/4 pixels above threshold (common)
        ], dtype=np.float32)

        # Biomass > 6.0 to count, occurrence ≥ 50% to be common
        calc = CommonSpecies(occurrence_threshold=0.5, biomass_threshold=6.0)
        result = calc.calculate(data)

        # Species 1: 4/4 pixels above threshold (common)
        # Species 2: 2/4 pixels above threshold = 50% (common)
        # Species 3: 0/4 pixels above threshold (not common)
        # Pixels with common species: [2, 2, 1, 1]
        expected = np.array([[2, 2, 1, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_common_species_exclude_total_layer(self):
        """Test that common species excludes total layer."""
        data = np.array([
            [[100, 100]],  # Total - should be excluded
            [[50, 50]],    # Species 1: 2/2 pixels (common)
            [[25, 0]]      # Species 2: 1/2 pixels = 50% (not common with 60% threshold)
        ], dtype=np.float32)

        calc = CommonSpecies(occurrence_threshold=0.6, biomass_threshold=0.0)
        result = calc.calculate(data)

        # Only species 1 is common
        expected = np.array([[1, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_common_species_none_common(self):
        """Test common species when no species meet threshold."""
        data = np.array([
            [[30, 20, 10, 0]],  # Total
            [[15, 0, 0, 0]],    # Species 1: 1/4 pixels = 25%
            [[15, 20, 0, 0]],   # Species 2: 2/4 pixels = 50%
            [[0, 0, 10, 0]]     # Species 3: 1/4 pixels = 25%
        ], dtype=np.float32)

        # Require 60% occurrence to be common
        calc = CommonSpecies(occurrence_threshold=0.6, biomass_threshold=0.0)
        result = calc.calculate(data)

        # No species meet 60% threshold
        expected = np.zeros((1, 4), dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_common_species_all_common(self):
        """Test common species when all species are common."""
        data = np.array([
            [[60, 60, 60]],  # Total
            [[20, 20, 20]],  # Species 1: everywhere
            [[20, 20, 20]],  # Species 2: everywhere
            [[20, 20, 20]]   # Species 3: everywhere
        ], dtype=np.float32)

        calc = CommonSpecies(occurrence_threshold=0.5, biomass_threshold=0.0)
        result = calc.calculate(data)

        # All 3 species are common
        expected = np.full((1, 3), 3, dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_common_species_single_species_layer(self):
        """Test common species with single species (no total layer)."""
        data = np.array([
            [[10, 15, 20, 0]]  # Single species in 3/4 pixels
        ], dtype=np.float32)

        calc = CommonSpecies(occurrence_threshold=0.7, biomass_threshold=0.0)
        result = calc.calculate(data)

        # Species present in 3/4 = 75% > 70% threshold (common)
        expected = np.array([[1, 1, 1, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_common_species_validation(self):
        """Test data validation for common species."""
        calc = CommonSpecies()

        # Valid 3D array
        valid_data = np.zeros((3, 10, 10))
        assert calc.validate_data(valid_data) is True

        # Invalid 2D array
        invalid_2d = np.zeros((10, 10))
        assert calc.validate_data(invalid_2d) is False

        # Empty array
        empty_data = np.zeros((0, 10, 10))
        assert calc.validate_data(empty_data) is False

    def test_common_species_kwargs_override(self):
        """Test kwargs override for common species parameters."""
        data = np.array([
            [[40, 30]],  # Total
            [[20, 0]],   # Species 1: 1/2 pixels = 50%
            [[20, 30]]   # Species 2: 2/2 pixels = 100%
        ], dtype=np.float32)

        calc = CommonSpecies(occurrence_threshold=0.8, biomass_threshold=0.0)

        # Override to lower threshold so species 1 becomes common
        result = calc.calculate(data, occurrence_threshold=0.4)
        expected = np.array([[2, 1]], dtype=np.uint8)  # Both common at pixel 1, only species 2 at pixel 2
        np.testing.assert_array_equal(result, expected)

        # Override biomass threshold
        result = calc.calculate(data, biomass_threshold=25.0)
        # Only species 2 at pixel 2 meets threshold, but 1/2 = 50% < 80% so not common
        expected = np.array([[0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_common_species_output_dtype(self):
        """Test common species output data type."""
        calc = CommonSpecies()
        assert calc.get_output_dtype() == np.uint8

    def test_common_species_metadata(self):
        """Test common species metadata."""
        calc = CommonSpecies(occurrence_threshold=0.15, biomass_threshold=5.0)
        metadata = calc.get_metadata()

        assert metadata['name'] == 'common_species'
        assert metadata['description'] == 'Count of common species per pixel'
        assert metadata['units'] == 'count'
        assert metadata['config']['occurrence_threshold'] == 0.15
        assert metadata['config']['biomass_threshold'] == 5.0


class TestSpeciesCalculationIntegration:
    """Integration tests using fixtures and real-world scenarios."""

    def test_with_sample_zarr_array(self, sample_zarr_array):
        """Test species calculations with sample zarr data from fixtures."""
        # Get the actual data array
        data = sample_zarr_array[:]

        # Test dominant species
        dominant_calc = DominantSpecies()
        dominant_result = dominant_calc.calculate(data)

        assert dominant_result.shape == (100, 100)
        assert dominant_result.dtype == np.uint8
        assert np.all(dominant_result >= 0)
        assert np.all(dominant_result <= 5)  # Max species index

        # Test species presence for different species
        presence_calc = SpeciesPresence(species_index=1, species_name="Dominant Oak")
        presence_result = presence_calc.calculate(data)

        assert presence_result.shape == (100, 100)
        assert presence_result.dtype == np.uint8
        assert np.all((presence_result == 0) | (presence_result == 1))

        # Test rare species
        rare_calc = RareSpecies(occurrence_threshold=0.05, biomass_threshold=1.0)
        rare_result = rare_calc.calculate(data)

        assert rare_result.shape == (100, 100)
        assert rare_result.dtype == np.uint8

        # Test common species
        common_calc = CommonSpecies(occurrence_threshold=0.3, biomass_threshold=1.0)
        common_result = common_calc.calculate(data)

        assert common_result.shape == (100, 100)
        assert common_result.dtype == np.uint8

    def test_with_empty_zarr_array(self, empty_zarr_array):
        """Test species calculations with empty zarr data."""
        data = empty_zarr_array[:]

        # Test dominant species with all zeros
        dominant_calc = DominantSpecies()
        dominant_result = dominant_calc.calculate(data)

        assert dominant_result.shape == (50, 50)
        np.testing.assert_array_equal(dominant_result, np.zeros((50, 50), dtype=np.uint8))

        # Test species presence with zeros
        presence_calc = SpeciesPresence(species_index=1, threshold=0.0)
        presence_result = presence_calc.calculate(data)

        np.testing.assert_array_equal(presence_result, np.zeros((50, 50), dtype=np.uint8))

        # Test rare species with zeros
        rare_calc = RareSpecies()
        rare_result = rare_calc.calculate(data)

        np.testing.assert_array_equal(rare_result, np.zeros((50, 50), dtype=np.uint8))

    def test_with_single_species_zarr(self, single_species_zarr):
        """Test species calculations with single species data."""
        data = single_species_zarr[:]

        # Test dominant species
        dominant_calc = DominantSpecies()
        dominant_result = dominant_calc.calculate(data)

        # Should be species 1 wherever there's biomass, 0 elsewhere
        expected = (data[1] > 0).astype(np.uint8)
        np.testing.assert_array_equal(dominant_result, expected)

        # Test species presence
        presence_calc = SpeciesPresence(species_index=1)
        presence_result = presence_calc.calculate(data)

        np.testing.assert_array_equal(presence_result, expected)

        # Test species dominance - should be 100% where species exists
        dominance_calc = SpeciesDominance(species_index=1)
        dominance_result = dominance_calc.calculate(data)

        # Calculate expected percentage
        total_pixels = dominance_result.size
        dominant_pixels = np.sum(expected)
        expected_percentage = (dominant_pixels / total_pixels) * 100.0

        np.testing.assert_array_almost_equal(dominance_result,
                                           np.full_like(dominance_result, expected_percentage))

    def test_edge_case_boundary_conditions(self):
        """Test edge cases and boundary conditions."""
        # Test with minimal data
        minimal_data = np.array([[[1]]], dtype=np.float32)

        # All calculations should work with 1x1 pixel
        calc_classes = [DominantSpecies, RareSpecies, CommonSpecies]
        for calc_class in calc_classes:
            calc = calc_class()
            result = calc.calculate(minimal_data)
            assert result.shape == (1, 1)

        # Species-specific calculations
        presence_calc = SpeciesPresence(species_index=0)
        presence_result = presence_calc.calculate(minimal_data)
        assert presence_result.shape == (1, 1)
        assert presence_result[0, 0] == 1

        dominance_calc = SpeciesDominance(species_index=0)
        with pytest.raises(ValueError):  # Index 0 not allowed
            dominance_calc.calculate(minimal_data)

    def test_calculation_consistency(self, sample_zarr_array):
        """Test consistency between related calculations."""
        data = sample_zarr_array[:]

        # Test that rare + common species counts are reasonable
        rare_calc = RareSpecies(occurrence_threshold=0.1)
        common_calc = CommonSpecies(occurrence_threshold=0.1)

        rare_result = rare_calc.calculate(data)
        common_result = common_calc.calculate(data)

        # At each pixel, rare + common should not exceed total species
        total_possible_species = data.shape[0] - 1  # Exclude total layer
        combined_counts = rare_result + common_result
        assert np.all(combined_counts <= total_possible_species)

    def test_metadata_completeness(self):
        """Test that all calculations provide complete metadata."""
        calculations = [
            DominantSpecies(),
            SpeciesPresence(species_index=1),
            SpeciesDominance(species_index=2, species_name="Test Species"),
            RareSpecies(),
            CommonSpecies()
        ]

        for calc in calculations:
            metadata = calc.get_metadata()

            # Required fields
            assert 'name' in metadata
            assert 'description' in metadata
            assert 'units' in metadata
            assert 'config' in metadata
            assert 'dtype' in metadata

            # Non-empty values
            assert len(metadata['name']) > 0
            assert len(metadata['description']) > 0
            assert len(metadata['units']) > 0
            assert isinstance(metadata['config'], dict)
            # dtype should be a valid numpy dtype
            assert isinstance(metadata['dtype'], (np.dtype, type)) or hasattr(metadata['dtype'], 'name')


class TestParameterHandlingAndConfig:
    """Test parameter handling and configuration across all species calculations."""

    def test_default_parameters(self):
        """Test that all calculations have sensible default parameters."""
        # DominantSpecies defaults
        calc = DominantSpecies()
        assert calc.config['exclude_total_layer'] is True
        assert calc.config['min_biomass'] == 0.0

        # SpeciesPresence defaults
        calc = SpeciesPresence(species_index=1)
        assert calc.config['species_index'] == 1
        assert calc.config['species_name'] is None
        assert calc.config['threshold'] == 0.0

        # SpeciesDominance defaults
        calc = SpeciesDominance(species_index=2)
        assert calc.config['species_index'] == 2
        assert calc.config['species_name'] is None

        # RareSpecies defaults
        calc = RareSpecies()
        assert calc.config['occurrence_threshold'] == 0.01
        assert calc.config['biomass_threshold'] == 0.0

        # CommonSpecies defaults
        calc = CommonSpecies()
        assert calc.config['occurrence_threshold'] == 0.10
        assert calc.config['biomass_threshold'] == 0.0

    def test_parameter_validation_types(self):
        """Test parameter type validation during initialization."""
        # Test valid parameter types
        DominantSpecies(exclude_total_layer=False, min_biomass=5.0)
        SpeciesPresence(species_index=3, species_name="Pine", threshold=1.0)
        SpeciesDominance(species_index=2, species_name="Oak")
        RareSpecies(occurrence_threshold=0.05, biomass_threshold=2.0)
        CommonSpecies(occurrence_threshold=0.20, biomass_threshold=3.0)

        # All should succeed without exceptions

    def test_configuration_inheritance(self):
        """Test that configuration is properly inherited and accessible."""
        calc = DominantSpecies(min_biomass=10.0, custom_param=42)

        assert calc.config['min_biomass'] == 10.0
        assert calc.config['custom_param'] == 42

        metadata = calc.get_metadata()
        assert metadata['config']['min_biomass'] == 10.0
        assert metadata['config']['custom_param'] == 42