"""Tests for ForestCalculation helper methods.

Tests the get_species_data and calculate_occurrence_frequency helper methods
added to the ForestCalculation base class for code deduplication.
"""

import pytest
import numpy as np

from gridfia.core.calculations.base import ForestCalculation
from gridfia.core.calculations.diversity import SpeciesRichness


class ConcreteCalculation(ForestCalculation):
    """Concrete implementation for testing abstract base class methods."""

    def calculate(self, biomass_data: np.ndarray, **kwargs) -> np.ndarray:
        """Simple calculation for testing."""
        return np.sum(biomass_data, axis=0)

    def validate_data(self, biomass_data: np.ndarray) -> bool:
        """Validate data for testing."""
        return biomass_data.ndim == 3 and biomass_data.shape[0] > 0


class TestGetSpeciesData:
    """Tests for the get_species_data helper method."""

    @pytest.fixture
    def calc(self):
        """Create a concrete calculation instance."""
        return ConcreteCalculation(name="test", description="Test calc", units="test")

    @pytest.fixture
    def biomass_3d(self):
        """Create a 3D biomass array with total layer at index 0."""
        # Shape: (4 layers, 3 height, 3 width)
        # Layer 0: total biomass (sum of others)
        # Layers 1-3: individual species
        data = np.array([
            [[30, 30, 30], [30, 30, 30], [30, 30, 30]],  # Total
            [[10, 10, 10], [10, 10, 10], [10, 10, 10]],  # Species 1
            [[10, 10, 10], [10, 10, 10], [10, 10, 10]],  # Species 2
            [[10, 10, 10], [10, 10, 10], [10, 10, 10]],  # Species 3
        ], dtype=np.float32)
        return data

    def test_exclude_total_true_returns_species_only(self, calc, biomass_3d):
        """Test that exclude_total=True returns only species layers."""
        result = calc.get_species_data(biomass_3d, exclude_total=True)

        assert result.shape == (3, 3, 3)  # 3 species, not 4
        np.testing.assert_array_equal(result, biomass_3d[1:])

    def test_exclude_total_false_returns_all_layers(self, calc, biomass_3d):
        """Test that exclude_total=False returns all layers."""
        result = calc.get_species_data(biomass_3d, exclude_total=False)

        assert result.shape == (4, 3, 3)  # All 4 layers
        np.testing.assert_array_equal(result, biomass_3d)

    def test_single_layer_returns_unchanged(self, calc):
        """Test that single-layer data is returned unchanged regardless of exclude_total."""
        single_layer = np.array([[[10, 20], [30, 40]]], dtype=np.float32)

        # Both should return the same since shape[0] = 1
        result_exclude = calc.get_species_data(single_layer, exclude_total=True)
        result_include = calc.get_species_data(single_layer, exclude_total=False)

        np.testing.assert_array_equal(result_exclude, single_layer)
        np.testing.assert_array_equal(result_include, single_layer)

    def test_two_layers_exclude_total(self, calc):
        """Test with exactly 2 layers (total + 1 species)."""
        two_layers = np.array([
            [[20, 20], [20, 20]],  # Total
            [[20, 20], [20, 20]],  # Species 1
        ], dtype=np.float32)

        result = calc.get_species_data(two_layers, exclude_total=True)

        assert result.shape == (1, 2, 2)  # Only species 1
        np.testing.assert_array_equal(result, two_layers[1:])

    def test_default_excludes_total(self, calc, biomass_3d):
        """Test that default behavior excludes total layer."""
        result = calc.get_species_data(biomass_3d)  # No exclude_total argument

        assert result.shape == (3, 3, 3)  # Should exclude total by default


class TestCalculateOccurrenceFrequency:
    """Tests for the calculate_occurrence_frequency static method."""

    def test_basic_occurrence_frequency(self):
        """Test basic occurrence frequency calculation."""
        # 2 species, 2x2 grid = 4 pixels
        # Species 1: present in 2/4 pixels = 0.5
        # Species 2: present in 4/4 pixels = 1.0
        data = np.array([
            [[10, 0], [10, 0]],   # Species 1: 2 pixels with biomass
            [[10, 10], [10, 10]], # Species 2: 4 pixels with biomass
        ], dtype=np.float32)

        freq = ForestCalculation.calculate_occurrence_frequency(data, biomass_threshold=0.0)

        np.testing.assert_array_almost_equal(freq, [0.5, 1.0])

    def test_occurrence_with_threshold(self):
        """Test occurrence frequency with biomass threshold."""
        # Species 1: values [5, 15, 5, 15] - 2 above threshold 10 = 0.5
        # Species 2: values [20, 20, 20, 20] - 4 above threshold 10 = 1.0
        data = np.array([
            [[5, 15], [5, 15]],   # Species 1
            [[20, 20], [20, 20]], # Species 2
        ], dtype=np.float32)

        freq = ForestCalculation.calculate_occurrence_frequency(data, biomass_threshold=10.0)

        np.testing.assert_array_almost_equal(freq, [0.5, 1.0])

    def test_all_zeros_returns_zero_frequency(self):
        """Test that all-zero data returns zero frequencies."""
        data = np.zeros((3, 4, 4), dtype=np.float32)

        freq = ForestCalculation.calculate_occurrence_frequency(data, biomass_threshold=0.0)

        np.testing.assert_array_equal(freq, [0.0, 0.0, 0.0])

    def test_all_present_returns_one(self):
        """Test that species present everywhere returns 1.0."""
        data = np.ones((2, 5, 5), dtype=np.float32) * 10

        freq = ForestCalculation.calculate_occurrence_frequency(data, biomass_threshold=0.0)

        np.testing.assert_array_almost_equal(freq, [1.0, 1.0])

    def test_single_species(self):
        """Test with single species."""
        data = np.array([[[10, 0, 10, 0]]], dtype=np.float32)  # 1 species, 1x4 grid

        freq = ForestCalculation.calculate_occurrence_frequency(data, biomass_threshold=0.0)

        assert len(freq) == 1
        np.testing.assert_almost_equal(freq[0], 0.5)

    def test_vectorized_performance(self):
        """Test that vectorized implementation handles large arrays efficiently."""
        # Large array: 100 species, 1000x1000 pixels
        np.random.seed(42)
        data = np.random.rand(100, 1000, 1000).astype(np.float32) * 100

        freq = ForestCalculation.calculate_occurrence_frequency(data, biomass_threshold=50.0)

        assert freq.shape == (100,)
        assert all(0 <= f <= 1 for f in freq)

    def test_default_threshold_is_zero(self):
        """Test that default threshold is 0.0."""
        data = np.array([
            [[0.001, 0], [0, 0]],  # Species 1: 1 pixel barely above 0
        ], dtype=np.float32)

        freq = ForestCalculation.calculate_occurrence_frequency(data)

        np.testing.assert_almost_equal(freq[0], 0.25)  # 1/4 pixels

    def test_rare_species_detection(self):
        """Test that occurrence frequency correctly identifies rare species."""
        # 3 species, 10x10 grid = 100 pixels
        data = np.zeros((3, 10, 10), dtype=np.float32)

        # Species 0: present in 1% of pixels (rare)
        data[0, 0, 0] = 10.0

        # Species 1: present in 5% of pixels (uncommon)
        data[1, 0, :5] = 10.0

        # Species 2: present in 50% of pixels (common)
        data[2, :5, :] = 10.0

        freq = ForestCalculation.calculate_occurrence_frequency(data, biomass_threshold=0.0)

        np.testing.assert_almost_equal(freq[0], 0.01)  # Rare
        np.testing.assert_almost_equal(freq[1], 0.05)  # Uncommon
        np.testing.assert_almost_equal(freq[2], 0.50)  # Common


class TestHelperIntegration:
    """Integration tests verifying helpers work with actual calculations."""

    def test_species_richness_uses_get_species_data_correctly(self):
        """Verify SpeciesRichness correctly uses get_species_data."""
        # 4 layers: total + 3 species
        data = np.array([
            [[30, 30], [30, 30]],  # Total (should be excluded)
            [[10, 0], [10, 0]],    # Species 1: 2 present
            [[10, 10], [0, 0]],    # Species 2: 2 present
            [[0, 10], [0, 10]],    # Species 3: 2 present
        ], dtype=np.float32)

        calc = SpeciesRichness(exclude_total_layer=True)
        result = calc.calculate(data)

        # Expected richness at each pixel:
        # [0,0]: species 1,2 present = 2
        # [0,1]: species 2,3 present = 2
        # [1,0]: species 1 present = 1
        # [1,1]: species 3 present = 1
        expected = np.array([[2, 2], [1, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_helpers_consistent_with_original_behavior(self):
        """Verify refactored code produces same results as original implementation."""
        np.random.seed(42)
        data = np.random.rand(5, 50, 50).astype(np.float32) * 100

        # Test get_species_data
        calc = ConcreteCalculation(name="test", description="", units="")
        species_data = calc.get_species_data(data, exclude_total=True)

        # Original implementation
        original_species_data = data[1:] if data.shape[0] > 1 else data

        np.testing.assert_array_equal(species_data, original_species_data)

        # Test occurrence frequency
        n_species, height, width = species_data.shape
        total_pixels = height * width

        # Original loop-based implementation
        threshold = 10.0
        original_freq = np.zeros(n_species)
        for i in range(n_species):
            original_freq[i] = np.sum(species_data[i] > threshold) / total_pixels

        # New vectorized implementation
        new_freq = ForestCalculation.calculate_occurrence_frequency(species_data, threshold)

        np.testing.assert_array_almost_equal(new_freq, original_freq)
