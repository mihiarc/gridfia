"""
Unit tests for diversity calculations.

Tests the Shannon diversity bug fix (Issue #2) and other diversity metrics.
"""

import pytest
import numpy as np
from gridfia.core.calculations.diversity import (
    ShannonDiversity,
    SimpsonDiversity,
    SpeciesRichness,
    Evenness
)


class TestShannonDiversity:
    """Test suite for Shannon diversity calculation."""

    def test_shannon_diversity_known_values(self):
        """Test Shannon diversity against known values."""
        # Example from ecological literature
        # 3 species with equal abundance (10 each) at pixel (0,0)
        # Expected Shannon: -3 * (1/3 * ln(1/3)) = ln(3) ≈ 1.0986
        # Shape: (n_species, height, width)
        data = np.array([
            [[10]],  # Species 1: 10 biomass at pixel (0,0)
            [[10]],  # Species 2: 10 biomass at pixel (0,0)
            [[10]]   # Species 3: 10 biomass at pixel (0,0)
        ])

        calc = ShannonDiversity(exclude_total_layer=False)
        result = calc.calculate(data)

        expected = -3 * (1/3 * np.log(1/3))
        np.testing.assert_almost_equal(result[0, 0], expected, decimal=6)

    def test_shannon_diversity_with_zeros(self):
        """Test Shannon diversity correctly handles zeros (Issue #2 fix)."""
        # Test data with zeros - should not add epsilon to non-zero values
        # Shape: (n_species, height, width)
        data = np.array([
            [[10]],  # Species 1
            [[20]],  # Species 2
            [[0]],   # Species 3 (zero)
            [[30]]   # Species 4
        ])

        calc = ShannonDiversity(exclude_total_layer=False)
        result = calc.calculate(data)

        # Manual calculation without epsilon bug
        total = 60
        p1, p2, p4 = 10/60, 20/60, 30/60
        expected = -(p1 * np.log(p1) + p2 * np.log(p2) + p4 * np.log(p4))

        np.testing.assert_almost_equal(result[0, 0], expected, decimal=6)

    def test_shannon_diversity_all_zeros(self):
        """Test Shannon diversity with all zero values."""
        data = np.array([
            [[0]],  # Species 1
            [[0]],  # Species 2
            [[0]]   # Species 3
        ])

        calc = ShannonDiversity(exclude_total_layer=False)
        result = calc.calculate(data)

        # Should return 0 for pixels with no biomass
        assert result[0, 0] == 0.0

    def test_shannon_diversity_single_species(self):
        """Test Shannon diversity with single species (should be 0)."""
        data = np.array([
            [[100]],  # Species 1
            [[0]],    # Species 2
            [[0]]     # Species 3
        ])

        calc = ShannonDiversity(exclude_total_layer=False)
        result = calc.calculate(data)

        # Single species has 0 diversity
        np.testing.assert_almost_equal(result[0, 0], 0.0, decimal=6)

    def test_shannon_diversity_base2(self):
        """Test Shannon diversity with base 2 logarithm."""
        # Two species with equal abundance
        data = np.array([
            [[50]],  # Species 1
            [[50]],  # Species 2
            [[0]]    # Species 3
        ])

        calc = ShannonDiversity(exclude_total_layer=False, base='2')
        result = calc.calculate(data)

        # With base 2: -2 * (0.5 * log2(0.5)) = 1.0 bit
        expected = -2 * (0.5 * np.log2(0.5))
        np.testing.assert_almost_equal(result[0, 0], expected, decimal=6)

    def test_shannon_diversity_exclude_total_layer(self):
        """Test Shannon diversity excluding first layer."""
        # First layer is total, next 3 are species
        data = np.array([
            [[60]],  # Total (should be excluded)
            [[10]],  # Species 1
            [[20]],  # Species 2
            [[30]]   # Species 3
        ])

        calc = ShannonDiversity(exclude_total_layer=True)
        result = calc.calculate(data)

        # Calculate expected from species layers only
        total = 60
        p1, p2, p3 = 10/60, 20/60, 30/60
        expected = -(p1 * np.log(p1) + p2 * np.log(p2) + p3 * np.log(p3))

        np.testing.assert_almost_equal(result[0, 0], expected, decimal=6)

    def test_shannon_diversity_2d_array(self):
        """Test Shannon diversity with 2D spatial data."""
        # 3 species, 2x2 spatial grid
        data = np.array([
            [[10, 20], [30, 0]],   # Species 1
            [[20, 10], [15, 0]],   # Species 2
            [[30, 5], [5, 100]]    # Species 3
        ])

        calc = ShannonDiversity(exclude_total_layer=False)
        result = calc.calculate(data)

        assert result.shape == (2, 2)

        # Check top-left pixel (10, 20, 30)
        total = 60
        p1, p2, p3 = 10/60, 20/60, 30/60
        expected_tl = -(p1 * np.log(p1) + p2 * np.log(p2) + p3 * np.log(p3))
        np.testing.assert_almost_equal(result[0, 0], expected_tl, decimal=6)

        # Check bottom-right pixel (0, 0, 100) - single species
        np.testing.assert_almost_equal(result[1, 1], 0.0, decimal=6)

    def test_shannon_diversity_no_epsilon_bias(self):
        """Verify the epsilon bug is fixed - no bias added to calculations."""
        # Create test case that would show epsilon bias
        data = np.array([
            [[10]],  # Species 1
            [[20]],  # Species 2
            [[30]],  # Species 3
            [[40]]   # Species 4
        ])

        calc = ShannonDiversity(exclude_total_layer=False)
        result = calc.calculate(data)

        # Calculate expected without any epsilon
        species_biomass = np.array([10, 20, 30, 40])
        proportions = species_biomass / species_biomass.sum()
        expected = -np.sum(proportions * np.log(proportions))

        # Should match closely (float32 precision)
        np.testing.assert_almost_equal(result[0, 0], expected, decimal=6)

    def test_reproduction_case_from_issue(self):
        """Test the exact reproduction case from Issue #2."""
        # Create test data from the issue
        data = np.array([
            [[10]],  # Species 1
            [[20]],  # Species 2
            [[0]],   # Species 3 (zero)
            [[30]]   # Species 4
        ])

        calc = ShannonDiversity(exclude_total_layer=False)
        result = calc.calculate(data)

        # Manual correct calculation from the issue
        species_biomass = np.array([10, 20, 0, 30])
        proportions = species_biomass / species_biomass.sum()
        valid_props = proportions[proportions > 0]
        shannon_correct = -np.sum(valid_props * np.log(valid_props))

        np.testing.assert_almost_equal(result[0, 0], shannon_correct, decimal=6)

        # Verify there's no epsilon-induced bias
        # The old buggy code would add epsilon to all values
        # This would create a small but measurable difference
        assert np.abs(result[0, 0] - shannon_correct) < 1e-6


class TestSimpsonDiversity:
    """Test suite for Simpson diversity calculation."""

    def test_simpson_diversity_known_values(self):
        """Test Simpson diversity against known values."""
        # 3 species with equal abundance
        data = np.array([
            [[10]],  # Species 1
            [[10]],  # Species 2
            [[10]]   # Species 3
        ])

        calc = SimpsonDiversity(exclude_total_layer=False, inverse=False)
        result = calc.calculate(data)

        # D = sum(pi^2) = 3 * (1/3)^2 = 1/3
        expected = 3 * (1/3)**2
        np.testing.assert_almost_equal(result[0, 0], expected, decimal=6)

    def test_simpson_diversity_inverse(self):
        """Test inverse Simpson diversity."""
        # 3 species with equal abundance
        data = np.array([
            [[10]],  # Species 1
            [[10]],  # Species 2
            [[10]]   # Species 3
        ])

        calc = SimpsonDiversity(exclude_total_layer=False, inverse=True)
        result = calc.calculate(data)

        # 1/D = 1/(1/3) = 3
        expected = 3.0
        np.testing.assert_almost_equal(result[0, 0], expected, decimal=6)

    def test_simpson_diversity_with_zeros(self):
        """Test Simpson diversity handles zeros correctly."""
        data = np.array([
            [[10]],  # Species 1
            [[0]],   # Species 2
            [[20]],  # Species 3
            [[0]]    # Species 4
        ])

        calc = SimpsonDiversity(exclude_total_layer=False, inverse=False)
        result = calc.calculate(data)

        # Only non-zero species contribute
        p1, p3 = 10/30, 20/30
        expected = p1**2 + p3**2
        np.testing.assert_almost_equal(result[0, 0], expected, decimal=6)


class TestSpeciesRichness:
    """Test suite for species richness calculation."""

    def test_species_richness_basic(self):
        """Test basic species richness counting."""
        data = np.array([
            [[10]],  # Species 1
            [[20]],  # Species 2
            [[0]],   # Species 3
            [[30]]   # Species 4
        ])

        calc = SpeciesRichness(biomass_threshold=0.0, exclude_total_layer=False)
        result = calc.calculate(data)

        # 3 species with biomass > 0
        assert result[0, 0] == 3

    def test_species_richness_with_threshold(self):
        """Test species richness with biomass threshold."""
        data = np.array([
            [[5]],   # Species 1
            [[15]],  # Species 2
            [[25]],  # Species 3
            [[35]]   # Species 4
        ])

        calc = SpeciesRichness(biomass_threshold=20, exclude_total_layer=False)
        result = calc.calculate(data)

        # Only 2 species above threshold
        assert result[0, 0] == 2

    def test_species_richness_exclude_total(self):
        """Test species richness excluding total layer."""
        data = np.array([
            [[60]],  # Total
            [[10]],  # Species 1
            [[0]],   # Species 2
            [[50]]   # Species 3
        ])

        calc = SpeciesRichness(exclude_total_layer=True)
        result = calc.calculate(data)

        # 2 species with biomass > 0 (excluding total)
        assert result[0, 0] == 2


class TestEvenness:
    """Test suite for Pielou's evenness calculation."""

    def test_evenness_equal_abundance(self):
        """Test evenness with equal species abundance (maximum evenness)."""
        # 3 species with equal abundance
        data = np.array([
            [[10]],  # Species 1
            [[10]],  # Species 2
            [[10]]   # Species 3
        ])

        calc = Evenness(exclude_total_layer=False)
        result = calc.calculate(data)

        # Maximum evenness = 1.0
        np.testing.assert_almost_equal(result[0, 0], 1.0, decimal=6)

    def test_evenness_unequal_abundance(self):
        """Test evenness with unequal species abundance."""
        # One dominant species
        data = np.array([
            [[90]],  # Species 1 (dominant)
            [[5]],   # Species 2
            [[5]]    # Species 3
        ])

        calc = Evenness(exclude_total_layer=False)
        result = calc.calculate(data)

        # Evenness should be less than 1
        assert 0 < result[0, 0] < 1

    def test_evenness_single_species(self):
        """Test evenness with single species (undefined)."""
        data = np.array([
            [[100]],  # Species 1
            [[0]],    # Species 2
            [[0]]     # Species 3
        ])

        calc = Evenness(exclude_total_layer=False)
        result = calc.calculate(data)

        # Evenness undefined for single species (returns 0)
        assert result[0, 0] == 0.0

    def test_evenness_two_species_equal(self):
        """Test evenness with two equal species."""
        data = np.array([
            [[50]],  # Species 1
            [[50]],  # Species 2
            [[0]]    # Species 3
        ])

        calc = Evenness(exclude_total_layer=False)
        result = calc.calculate(data)

        # Maximum evenness for 2 species
        np.testing.assert_almost_equal(result[0, 0], 1.0, decimal=6)


class TestDiversityValidation:
    """Test validation methods for diversity calculations."""

    def test_validate_3d_array(self):
        """Test that 3D arrays are accepted."""
        data = np.zeros((5, 10, 10))

        calc = ShannonDiversity()
        assert calc.validate_data(data) is True

    def test_validate_2d_array_rejected(self):
        """Test that 2D arrays are rejected."""
        data = np.zeros((10, 10))

        calc = ShannonDiversity()
        assert calc.validate_data(data) is False

    def test_validate_empty_array_rejected(self):
        """Test that empty arrays are rejected."""
        data = np.zeros((0, 10, 10))

        calc = ShannonDiversity()
        assert calc.validate_data(data) is False