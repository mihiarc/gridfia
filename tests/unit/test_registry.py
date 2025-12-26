"""
Unit tests for the CalculationRegistry and convenience functions.

Tests the registry pattern implementation for managing forest metric calculations,
including registration, retrieval, and group creation functionality.
"""

import pytest
import numpy as np
import logging
from typing import Dict, Any

from gridfia.core.calculations.registry import (
    CalculationRegistry,
    registry,
    register_calculation,
    get_calculation,
    list_calculations,
)
from gridfia.core.calculations.base import ForestCalculation
from gridfia.core.calculations.diversity import (
    ShannonDiversity,
    SimpsonDiversity,
    SpeciesRichness,
    Evenness,
)
from gridfia.core.calculations.biomass import (
    TotalBiomass,
    TotalBiomassComparison,
    SpeciesProportion,
    SpeciesPercentage,
    SpeciesGroupProportion,
    BiomassThreshold,
)
from gridfia.core.calculations.species import (
    DominantSpecies,
    SpeciesPresence,
    SpeciesDominance,
    RareSpecies,
    CommonSpecies,
)


# Test fixtures for custom calculations
class MockForestCalculation(ForestCalculation):
    """Mock calculation for testing registration."""

    def __init__(self, **kwargs):
        super().__init__(
            name="mock_calculation",
            description="A mock calculation for testing",
            units="test_units",
            **kwargs,
        )

    def calculate(self, biomass_data: np.ndarray, **kwargs) -> np.ndarray:
        """Return zeros with the same spatial shape as input."""
        return np.zeros(biomass_data.shape[1:], dtype=np.float32)

    def validate_data(self, biomass_data: np.ndarray) -> bool:
        """Validate that input is a 3D array."""
        return biomass_data.ndim == 3 and biomass_data.shape[0] > 0


class AnotherMockCalculation(ForestCalculation):
    """Another mock calculation for testing overwrites."""

    def __init__(self, custom_param: str = "default", **kwargs):
        super().__init__(
            name="another_mock",
            description="Another mock calculation",
            units="other_units",
            custom_param=custom_param,
            **kwargs,
        )
        self.custom_param = custom_param

    def calculate(self, biomass_data: np.ndarray, **kwargs) -> np.ndarray:
        """Return ones with the same spatial shape as input."""
        return np.ones(biomass_data.shape[1:], dtype=np.float32)

    def validate_data(self, biomass_data: np.ndarray) -> bool:
        """Validate that input is a 3D array."""
        return biomass_data.ndim == 3


class InvalidCalculation:
    """A class that does not inherit from ForestCalculation."""

    def calculate(self, data):
        return data


class TestCalculationRegistryInitialization:
    """Test suite for CalculationRegistry initialization."""

    def test_registry_initializes_with_default_calculations(self):
        """Test that a new registry contains all default calculations."""
        test_registry = CalculationRegistry()
        available_calculations = test_registry.list_calculations()

        # Check diversity calculations are registered
        assert "species_richness" in available_calculations
        assert "shannon_diversity" in available_calculations
        assert "simpson_diversity" in available_calculations
        assert "evenness" in available_calculations

        # Check biomass calculations are registered
        assert "total_biomass" in available_calculations
        assert "total_biomass_comparison" in available_calculations
        assert "species_proportion" in available_calculations
        assert "species_percentage" in available_calculations
        assert "species_group_proportion" in available_calculations
        assert "biomass_threshold" in available_calculations

        # Check species calculations are registered
        assert "dominant_species" in available_calculations
        assert "species_presence" in available_calculations
        assert "species_dominance" in available_calculations
        assert "rare_species" in available_calculations
        assert "common_species" in available_calculations

    def test_registry_initializes_with_correct_count(self):
        """Test that the registry has exactly 15 default calculations."""
        test_registry = CalculationRegistry()
        available_calculations = test_registry.list_calculations()

        # 4 diversity + 6 biomass + 5 species = 15 calculations
        assert len(available_calculations) == 15

    def test_registry_initializes_empty_internal_dict_before_defaults(self):
        """Test that registry properly initializes internal storage."""
        test_registry = CalculationRegistry()

        # The internal dict should be populated
        assert hasattr(test_registry, "_calculations")
        assert isinstance(test_registry._calculations, dict)
        assert len(test_registry._calculations) == 15

    def test_multiple_registries_are_independent(self):
        """Test that multiple registry instances are independent."""
        registry_one = CalculationRegistry()
        registry_two = CalculationRegistry()

        # Register custom calculation in registry_one only
        registry_one.register("custom_test", MockForestCalculation)

        # registry_one should have the custom calculation
        assert "custom_test" in registry_one.list_calculations()

        # registry_two should not have the custom calculation
        assert "custom_test" not in registry_two.list_calculations()


class TestCalculationRegistryRegister:
    """Test suite for the register() method."""

    def test_register_valid_calculation(self):
        """Test registering a valid calculation class."""
        test_registry = CalculationRegistry()
        initial_count = len(test_registry.list_calculations())

        test_registry.register("mock_calc", MockForestCalculation)

        assert "mock_calc" in test_registry.list_calculations()
        assert len(test_registry.list_calculations()) == initial_count + 1

    def test_register_with_existing_name_overwrites(self, caplog):
        """Test that registering with an existing name logs a warning."""
        test_registry = CalculationRegistry()

        with caplog.at_level(logging.WARNING):
            test_registry.register("species_richness", MockForestCalculation)

        assert "Overwriting existing calculation: species_richness" in caplog.text

        # Verify it was overwritten
        calc = test_registry.get("species_richness")
        assert isinstance(calc, MockForestCalculation)

    def test_register_invalid_class_raises_error(self):
        """Test that registering a non-ForestCalculation class raises ValueError."""
        test_registry = CalculationRegistry()

        with pytest.raises(ValueError) as excinfo:
            test_registry.register("invalid", InvalidCalculation)

        assert "must be a subclass of ForestCalculation" in str(excinfo.value)

    def test_register_non_class_raises_error(self):
        """Test that registering a non-class object raises an error."""
        test_registry = CalculationRegistry()

        with pytest.raises(TypeError):
            test_registry.register("not_a_class", "string_value")

    def test_register_logs_debug_message(self, caplog):
        """Test that successful registration logs a debug message."""
        test_registry = CalculationRegistry()

        with caplog.at_level(logging.DEBUG):
            test_registry.register("debug_test_calc", MockForestCalculation)

        assert "Registered calculation: debug_test_calc" in caplog.text


class TestCalculationRegistryUnregister:
    """Test suite for the unregister() method."""

    def test_unregister_existing_calculation(self):
        """Test unregistering an existing calculation."""
        test_registry = CalculationRegistry()
        test_registry.register("to_remove", MockForestCalculation)

        assert "to_remove" in test_registry.list_calculations()

        test_registry.unregister("to_remove")

        assert "to_remove" not in test_registry.list_calculations()

    def test_unregister_default_calculation(self):
        """Test unregistering a default calculation."""
        test_registry = CalculationRegistry()

        assert "species_richness" in test_registry.list_calculations()

        test_registry.unregister("species_richness")

        assert "species_richness" not in test_registry.list_calculations()

    def test_unregister_nonexistent_calculation_logs_warning(self, caplog):
        """Test that unregistering a nonexistent calculation logs a warning."""
        test_registry = CalculationRegistry()

        with caplog.at_level(logging.WARNING):
            test_registry.unregister("nonexistent_calculation")

        assert "Calculation not found: nonexistent_calculation" in caplog.text

    def test_unregister_logs_debug_on_success(self, caplog):
        """Test that successful unregistration logs a debug message."""
        test_registry = CalculationRegistry()
        test_registry.register("unregister_test", MockForestCalculation)

        with caplog.at_level(logging.DEBUG):
            test_registry.unregister("unregister_test")

        assert "Unregistered calculation: unregister_test" in caplog.text


class TestCalculationRegistryGet:
    """Test suite for the get() method."""

    def test_get_default_calculation(self):
        """Test getting a default calculation instance."""
        test_registry = CalculationRegistry()

        calc = test_registry.get("species_richness")

        assert isinstance(calc, SpeciesRichness)

    def test_get_calculation_with_kwargs(self):
        """Test getting a calculation instance with custom parameters."""
        test_registry = CalculationRegistry()

        calc = test_registry.get("species_richness", biomass_threshold=10.0)

        assert isinstance(calc, SpeciesRichness)
        # The config should contain the threshold parameter
        assert calc.config.get("biomass_threshold") == 10.0

    def test_get_custom_calculation_with_kwargs(self):
        """Test getting a custom calculation with parameters."""
        test_registry = CalculationRegistry()
        test_registry.register("another_mock", AnotherMockCalculation)

        calc = test_registry.get("another_mock", custom_param="custom_value")

        assert isinstance(calc, AnotherMockCalculation)
        assert calc.custom_param == "custom_value"

    def test_get_unknown_calculation_raises_error(self):
        """Test that getting an unknown calculation raises ValueError."""
        test_registry = CalculationRegistry()

        with pytest.raises(ValueError) as excinfo:
            test_registry.get("nonexistent_calculation")

        assert "Unknown calculation: nonexistent_calculation" in str(excinfo.value)

    def test_get_returns_new_instance_each_time(self):
        """Test that get() returns a new instance each call."""
        test_registry = CalculationRegistry()

        calc_one = test_registry.get("species_richness")
        calc_two = test_registry.get("species_richness")

        assert calc_one is not calc_two
        assert type(calc_one) == type(calc_two)

    def test_get_calculations_without_required_params(self):
        """Test that calculations without required params can be instantiated."""
        test_registry = CalculationRegistry()

        # These calculations have all optional params with defaults
        no_required_params = [
            "species_richness",
            "shannon_diversity",
            "simpson_diversity",
            "evenness",
            "total_biomass",
            "total_biomass_comparison",
            "dominant_species",
            "rare_species",
            "common_species",
        ]

        for calc_name in no_required_params:
            calc = test_registry.get(calc_name)
            assert isinstance(calc, ForestCalculation)
            assert calc.name is not None
            assert calc.description is not None
            assert calc.units is not None

    def test_get_calculations_with_required_params(self):
        """Test calculations that require parameters."""
        test_registry = CalculationRegistry()

        # species_proportion requires species_index
        calc = test_registry.get("species_proportion", species_index=1)
        assert isinstance(calc, SpeciesProportion)

        # species_percentage requires species_index
        calc = test_registry.get("species_percentage", species_index=2)
        assert isinstance(calc, SpeciesPercentage)

        # species_group_proportion requires species_indices and group_name
        calc = test_registry.get(
            "species_group_proportion",
            species_indices=[1, 2],
            group_name="test_group",
        )
        assert isinstance(calc, SpeciesGroupProportion)

        # biomass_threshold requires threshold
        calc = test_registry.get("biomass_threshold", threshold=50.0)
        assert isinstance(calc, BiomassThreshold)

        # species_presence requires species_index
        calc = test_registry.get("species_presence", species_index=1)
        assert isinstance(calc, SpeciesPresence)

        # species_dominance requires species_index
        calc = test_registry.get("species_dominance", species_index=1)
        assert isinstance(calc, SpeciesDominance)


class TestCalculationRegistryGetCalculationInfo:
    """Test suite for the get_calculation_info() method."""

    def test_get_calculation_info_returns_metadata(self):
        """Test that get_calculation_info returns calculation metadata."""
        test_registry = CalculationRegistry()

        info = test_registry.get_calculation_info("species_richness")

        assert info is not None
        assert "name" in info
        assert "description" in info
        assert "units" in info
        assert "config" in info
        assert "dtype" in info

    def test_get_calculation_info_correct_values(self):
        """Test that get_calculation_info returns correct metadata values."""
        test_registry = CalculationRegistry()
        test_registry.register("mock_calc", MockForestCalculation)

        info = test_registry.get_calculation_info("mock_calc")

        assert info["name"] == "mock_calculation"
        assert info["description"] == "A mock calculation for testing"
        assert info["units"] == "test_units"

    def test_get_calculation_info_unknown_returns_none(self):
        """Test that get_calculation_info returns None for unknown calculations."""
        test_registry = CalculationRegistry()

        info = test_registry.get_calculation_info("nonexistent")

        assert info is None

    def test_get_calculation_info_includes_dtype(self):
        """Test that get_calculation_info includes output dtype."""
        test_registry = CalculationRegistry()

        info = test_registry.get_calculation_info("total_biomass")

        assert "dtype" in info
        # TotalBiomass should return float32
        assert info["dtype"] == np.float32


class TestCalculationRegistryGetAllInfo:
    """Test suite for the get_all_info() method."""

    def test_get_all_info_returns_info_for_instantiable_calculations(self):
        """Test that get_all_info returns info for calculations that can be instantiated without params."""
        test_registry = CalculationRegistry()

        # Unregister calculations that require parameters for this test
        # as get_all_info() calls get() which requires those params
        calcs_with_required_params = [
            "species_proportion",
            "species_percentage",
            "species_group_proportion",
            "biomass_threshold",
            "species_presence",
            "species_dominance",
        ]
        for calc_name in calcs_with_required_params:
            test_registry.unregister(calc_name)

        all_info = test_registry.get_all_info()

        # Should have 15 - 6 = 9 calculations
        assert len(all_info) == 9
        for calc_name in test_registry.list_calculations():
            assert calc_name in all_info

    def test_get_all_info_each_entry_has_metadata(self):
        """Test that each entry in get_all_info has complete metadata."""
        test_registry = CalculationRegistry()

        # Unregister calculations that require parameters
        calcs_with_required_params = [
            "species_proportion",
            "species_percentage",
            "species_group_proportion",
            "biomass_threshold",
            "species_presence",
            "species_dominance",
        ]
        for calc_name in calcs_with_required_params:
            test_registry.unregister(calc_name)

        all_info = test_registry.get_all_info()

        for calc_name, info in all_info.items():
            assert "name" in info, f"{calc_name} missing 'name'"
            assert "description" in info, f"{calc_name} missing 'description'"
            assert "units" in info, f"{calc_name} missing 'units'"
            assert "config" in info, f"{calc_name} missing 'config'"
            assert "dtype" in info, f"{calc_name} missing 'dtype'"

    def test_get_all_info_with_custom_calculation(self):
        """Test get_all_info includes custom registered calculations."""
        test_registry = CalculationRegistry()

        # Unregister calculations that require parameters
        calcs_with_required_params = [
            "species_proportion",
            "species_percentage",
            "species_group_proportion",
            "biomass_threshold",
            "species_presence",
            "species_dominance",
        ]
        for calc_name in calcs_with_required_params:
            test_registry.unregister(calc_name)

        test_registry.register("custom_mock", MockForestCalculation)

        all_info = test_registry.get_all_info()

        assert "custom_mock" in all_info
        assert all_info["custom_mock"]["name"] == "mock_calculation"


class TestCalculationRegistryListCalculations:
    """Test suite for the list_calculations() method."""

    def test_list_calculations_returns_sorted_list(self):
        """Test that list_calculations returns a sorted list."""
        test_registry = CalculationRegistry()

        calculations = test_registry.list_calculations()

        assert calculations == sorted(calculations)

    def test_list_calculations_returns_list(self):
        """Test that list_calculations returns a list type."""
        test_registry = CalculationRegistry()

        calculations = test_registry.list_calculations()

        assert isinstance(calculations, list)

    def test_list_calculations_contains_strings(self):
        """Test that list_calculations returns string names."""
        test_registry = CalculationRegistry()

        calculations = test_registry.list_calculations()

        for calc_name in calculations:
            assert isinstance(calc_name, str)

    def test_list_calculations_after_registration(self):
        """Test list_calculations reflects new registrations."""
        test_registry = CalculationRegistry()
        initial_calculations = test_registry.list_calculations()

        test_registry.register("zzz_last_calculation", MockForestCalculation)

        updated_calculations = test_registry.list_calculations()

        assert len(updated_calculations) == len(initial_calculations) + 1
        assert "zzz_last_calculation" in updated_calculations
        # Should still be sorted
        assert updated_calculations == sorted(updated_calculations)

    def test_list_calculations_after_unregistration(self):
        """Test list_calculations reflects unregistrations."""
        test_registry = CalculationRegistry()
        initial_count = len(test_registry.list_calculations())

        test_registry.unregister("species_richness")

        updated_calculations = test_registry.list_calculations()

        assert len(updated_calculations) == initial_count - 1
        assert "species_richness" not in updated_calculations


class TestCalculationRegistryCreateCalculationGroup:
    """Test suite for the create_calculation_group() method."""

    def test_create_calculation_group_basic(self):
        """Test creating a group of calculations from configuration."""
        test_registry = CalculationRegistry()
        calc_configs = [
            {"name": "species_richness"},
            {"name": "shannon_diversity"},
            {"name": "total_biomass"},
        ]

        instances = test_registry.create_calculation_group(calc_configs)

        assert len(instances) == 3
        assert isinstance(instances[0], SpeciesRichness)
        assert isinstance(instances[1], ShannonDiversity)
        assert isinstance(instances[2], TotalBiomass)

    def test_create_calculation_group_with_parameters(self):
        """Test creating calculations with custom parameters."""
        test_registry = CalculationRegistry()
        calc_configs = [
            {"name": "species_richness", "biomass_threshold": 5.0},
            {"name": "shannon_diversity", "base": "2"},
        ]

        instances = test_registry.create_calculation_group(calc_configs)

        assert len(instances) == 2
        assert instances[0].config.get("biomass_threshold") == 5.0
        assert instances[1].config.get("base") == "2"

    def test_create_calculation_group_empty_list(self):
        """Test creating a group from an empty list."""
        test_registry = CalculationRegistry()

        instances = test_registry.create_calculation_group([])

        assert instances == []

    def test_create_calculation_group_skips_missing_name(self, caplog):
        """Test that configs without 'name' are skipped with a warning."""
        test_registry = CalculationRegistry()
        calc_configs = [
            {"name": "species_richness"},
            {"some_param": "value"},  # Missing 'name'
            {"name": "total_biomass"},
        ]

        with caplog.at_level(logging.WARNING):
            instances = test_registry.create_calculation_group(calc_configs)

        assert len(instances) == 2
        assert "Skipping calculation without name" in caplog.text

    def test_create_calculation_group_handles_unknown_calculation(self, caplog):
        """Test that unknown calculations are logged and skipped."""
        test_registry = CalculationRegistry()
        calc_configs = [
            {"name": "species_richness"},
            {"name": "nonexistent_calculation"},
            {"name": "total_biomass"},
        ]

        with caplog.at_level(logging.ERROR):
            instances = test_registry.create_calculation_group(calc_configs)

        assert len(instances) == 2
        assert "Failed to create calculation nonexistent_calculation" in caplog.text

    def test_create_calculation_group_handles_invalid_params(self, caplog):
        """Test that invalid parameters are logged and the calculation is skipped."""
        test_registry = CalculationRegistry()
        test_registry.register("strict_mock", AnotherMockCalculation)

        calc_configs = [
            {"name": "species_richness"},
            {"name": "strict_mock", "invalid_constructor_param": 123},
        ]

        # This should not raise but should log an error if the param causes an issue
        with caplog.at_level(logging.ERROR):
            instances = test_registry.create_calculation_group(calc_configs)

        # The mock accepts **kwargs so it should still work
        assert len(instances) == 2

    def test_create_calculation_group_preserves_order(self):
        """Test that calculation order is preserved."""
        test_registry = CalculationRegistry()
        calc_configs = [
            {"name": "total_biomass"},
            {"name": "species_richness"},
            {"name": "shannon_diversity"},
        ]

        instances = test_registry.create_calculation_group(calc_configs)

        assert isinstance(instances[0], TotalBiomass)
        assert isinstance(instances[1], SpeciesRichness)
        assert isinstance(instances[2], ShannonDiversity)


class TestGlobalRegistry:
    """Test suite for the global registry instance."""

    def test_global_registry_exists(self):
        """Test that the global registry instance exists."""
        assert registry is not None
        assert isinstance(registry, CalculationRegistry)

    def test_global_registry_has_default_calculations(self):
        """Test that the global registry has all default calculations."""
        calculations = registry.list_calculations()

        assert "species_richness" in calculations
        assert "shannon_diversity" in calculations
        assert "total_biomass" in calculations
        assert "dominant_species" in calculations

    def test_global_registry_is_singleton_like(self):
        """Test that importing registry always gives the same instance."""
        from gridfia.core.calculations.registry import registry as registry_reimport

        assert registry is registry_reimport

    def test_global_registry_modifications_persist(self):
        """Test that modifications to global registry persist within module."""
        # Register a custom calculation
        unique_name = "test_global_persistence_calc"
        registry.register(unique_name, MockForestCalculation)

        # Verify it's accessible
        assert unique_name in registry.list_calculations()

        # Clean up
        registry.unregister(unique_name)
        assert unique_name not in registry.list_calculations()


class TestConvenienceFunctions:
    """Test suite for convenience functions."""

    def test_register_calculation_function(self):
        """Test the register_calculation convenience function."""
        unique_name = "convenience_test_calc"

        try:
            register_calculation(unique_name, MockForestCalculation)

            assert unique_name in registry.list_calculations()
        finally:
            # Clean up
            registry.unregister(unique_name)

    def test_get_calculation_function(self):
        """Test the get_calculation convenience function."""
        calc = get_calculation("species_richness")

        assert isinstance(calc, SpeciesRichness)

    def test_get_calculation_function_with_params(self):
        """Test get_calculation with custom parameters."""
        calc = get_calculation("species_richness", biomass_threshold=15.0)

        assert isinstance(calc, SpeciesRichness)
        assert calc.config.get("biomass_threshold") == 15.0

    def test_get_calculation_function_unknown_raises_error(self):
        """Test that get_calculation raises error for unknown calculation."""
        with pytest.raises(ValueError) as excinfo:
            get_calculation("unknown_calculation")

        assert "Unknown calculation: unknown_calculation" in str(excinfo.value)

    def test_list_calculations_function(self):
        """Test the list_calculations convenience function."""
        calculations = list_calculations()

        assert isinstance(calculations, list)
        assert "species_richness" in calculations
        assert "shannon_diversity" in calculations
        assert calculations == sorted(calculations)

    def test_convenience_functions_use_global_registry(self):
        """Test that convenience functions operate on the global registry."""
        unique_name = "convenience_global_test"

        try:
            # Register via convenience function
            register_calculation(unique_name, MockForestCalculation)

            # Should be visible in global registry
            assert unique_name in registry.list_calculations()

            # Should be gettable via convenience function
            calc = get_calculation(unique_name)
            assert isinstance(calc, MockForestCalculation)

            # Should appear in list
            assert unique_name in list_calculations()
        finally:
            # Clean up
            registry.unregister(unique_name)


class TestCalculationRegistryEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    def test_register_same_calculation_multiple_times(self, caplog):
        """Test registering the same calculation class under different names."""
        test_registry = CalculationRegistry()

        test_registry.register("mock_v1", MockForestCalculation)
        test_registry.register("mock_v2", MockForestCalculation)

        # Both should be registered
        assert "mock_v1" in test_registry.list_calculations()
        assert "mock_v2" in test_registry.list_calculations()

        # Both should return same class type
        calc_v1 = test_registry.get("mock_v1")
        calc_v2 = test_registry.get("mock_v2")
        assert type(calc_v1) == type(calc_v2)

    def test_register_empty_name(self):
        """Test registering with an empty name."""
        test_registry = CalculationRegistry()

        # Should work but not recommended
        test_registry.register("", MockForestCalculation)

        assert "" in test_registry.list_calculations()

    def test_register_name_with_special_characters(self):
        """Test registering with special characters in name."""
        test_registry = CalculationRegistry()

        test_registry.register("special-name_v2.0", MockForestCalculation)

        assert "special-name_v2.0" in test_registry.list_calculations()
        calc = test_registry.get("special-name_v2.0")
        assert isinstance(calc, MockForestCalculation)

    def test_get_calculation_info_after_unregister(self):
        """Test get_calculation_info returns None after unregistration."""
        test_registry = CalculationRegistry()
        test_registry.register("temp_calc", MockForestCalculation)

        # Should return info before unregister
        info_before = test_registry.get_calculation_info("temp_calc")
        assert info_before is not None

        test_registry.unregister("temp_calc")

        # Should return None after unregister
        info_after = test_registry.get_calculation_info("temp_calc")
        assert info_after is None

    def test_create_calculation_group_with_duplicate_names(self):
        """Test creating a group with duplicate calculation names."""
        test_registry = CalculationRegistry()
        calc_configs = [
            {"name": "species_richness"},
            {"name": "species_richness"},
            {"name": "species_richness"},
        ]

        instances = test_registry.create_calculation_group(calc_configs)

        assert len(instances) == 3
        for instance in instances:
            assert isinstance(instance, SpeciesRichness)

    def test_unregister_then_reregister(self):
        """Test unregistering and re-registering a calculation."""
        test_registry = CalculationRegistry()

        # Unregister default
        test_registry.unregister("species_richness")
        assert "species_richness" not in test_registry.list_calculations()

        # Re-register with different class
        test_registry.register("species_richness", MockForestCalculation)

        # Should now return the mock
        calc = test_registry.get("species_richness")
        assert isinstance(calc, MockForestCalculation)


class TestCalculationRegistryIntegrationWithCalculations:
    """Integration tests verifying registry works with actual calculation classes."""

    def test_all_diversity_calculations_instantiate(self):
        """Test all diversity calculations can be instantiated."""
        test_registry = CalculationRegistry()

        diversity_calcs = [
            "species_richness",
            "shannon_diversity",
            "simpson_diversity",
            "evenness",
        ]

        for calc_name in diversity_calcs:
            calc = test_registry.get(calc_name)
            assert isinstance(calc, ForestCalculation)
            assert calc.validate_data(np.zeros((3, 10, 10)))

    def test_all_biomass_calculations_instantiate(self):
        """Test all biomass calculations can be instantiated with appropriate params."""
        test_registry = CalculationRegistry()

        # Calculations without required params
        simple_biomass_calcs = [
            "total_biomass",
            "total_biomass_comparison",
        ]

        for calc_name in simple_biomass_calcs:
            calc = test_registry.get(calc_name)
            assert isinstance(calc, ForestCalculation)

        # Calculations with required params
        calc = test_registry.get("species_proportion", species_index=1)
        assert isinstance(calc, ForestCalculation)

        calc = test_registry.get("species_percentage", species_index=1)
        assert isinstance(calc, ForestCalculation)

        calc = test_registry.get(
            "species_group_proportion",
            species_indices=[1, 2],
            group_name="test",
        )
        assert isinstance(calc, ForestCalculation)

        calc = test_registry.get("biomass_threshold", threshold=50.0)
        assert isinstance(calc, ForestCalculation)

    def test_all_species_calculations_instantiate(self):
        """Test all species calculations can be instantiated with appropriate params."""
        test_registry = CalculationRegistry()

        # Calculations without required params
        simple_species_calcs = [
            "dominant_species",
            "rare_species",
            "common_species",
        ]

        for calc_name in simple_species_calcs:
            calc = test_registry.get(calc_name)
            assert isinstance(calc, ForestCalculation)

        # Calculations with required params
        calc = test_registry.get("species_presence", species_index=1)
        assert isinstance(calc, ForestCalculation)

        calc = test_registry.get("species_dominance", species_index=1)
        assert isinstance(calc, ForestCalculation)

    def test_calculation_metadata_for_parameterless_calculations(self):
        """Test that calculations without required params have complete metadata."""
        test_registry = CalculationRegistry()

        # Only test calculations that can be instantiated without params
        parameterless_calcs = [
            "species_richness",
            "shannon_diversity",
            "simpson_diversity",
            "evenness",
            "total_biomass",
            "total_biomass_comparison",
            "dominant_species",
            "rare_species",
            "common_species",
        ]

        for calc_name in parameterless_calcs:
            info = test_registry.get_calculation_info(calc_name)

            assert info is not None, f"{calc_name} returned None metadata"
            assert info["name"], f"{calc_name} has empty name"
            assert info["description"], f"{calc_name} has empty description"
            assert info["units"] is not None, f"{calc_name} has no units"
            assert "dtype" in info, f"{calc_name} missing dtype"

    def test_calculation_metadata_for_parameterized_calculations(self):
        """Test that calculations with required params have complete metadata when instantiated."""
        test_registry = CalculationRegistry()

        # Test species_proportion
        calc = test_registry.get("species_proportion", species_index=1)
        metadata = calc.get_metadata()
        assert metadata["name"] is not None
        assert metadata["description"] is not None
        assert "dtype" in metadata

        # Test species_group_proportion
        calc = test_registry.get(
            "species_group_proportion",
            species_indices=[1, 2],
            group_name="test_group",
        )
        metadata = calc.get_metadata()
        assert metadata["name"] is not None
        assert "test_group" in metadata["name"].lower() or "test" in metadata["name"]

        # Test biomass_threshold
        calc = test_registry.get("biomass_threshold", threshold=100.0)
        metadata = calc.get_metadata()
        assert metadata["units"] == "boolean"
