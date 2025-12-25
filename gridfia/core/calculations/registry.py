"""
Calculation registry for managing forest metric calculations.

This module provides a registry pattern for discovering and managing
available calculations dynamically.
"""

import logging
from typing import Dict, List, Optional, Type, Any

from .base import ForestCalculation
from .diversity import (
    SpeciesRichness, ShannonDiversity, SimpsonDiversity, Evenness
)
from .biomass import (
    TotalBiomass, TotalBiomassComparison, SpeciesProportion,
    SpeciesPercentage, SpeciesGroupProportion, BiomassThreshold
)
from .species import (
    DominantSpecies, SpeciesPresence, SpeciesDominance,
    RareSpecies, CommonSpecies
)

logger = logging.getLogger(__name__)


class CalculationRegistry:
    """Registry for managing available forest calculations."""
    
    def __init__(self):
        """Initialize the calculation registry."""
        self._calculations: Dict[str, Type[ForestCalculation]] = {}
        self._register_default_calculations()
    
    def _register_default_calculations(self):
        """Register all default calculations."""
        # Diversity calculations
        self.register("species_richness", SpeciesRichness)
        self.register("shannon_diversity", ShannonDiversity)
        self.register("simpson_diversity", SimpsonDiversity)
        self.register("evenness", Evenness)
        
        # Biomass calculations
        self.register("total_biomass", TotalBiomass)
        self.register("total_biomass_comparison", TotalBiomassComparison)
        self.register("species_proportion", SpeciesProportion)
        self.register("species_percentage", SpeciesPercentage)
        self.register("species_group_proportion", SpeciesGroupProportion)
        self.register("biomass_threshold", BiomassThreshold)
        
        # Species calculations
        self.register("dominant_species", DominantSpecies)
        self.register("species_presence", SpeciesPresence)
        self.register("species_dominance", SpeciesDominance)
        self.register("rare_species", RareSpecies)
        self.register("common_species", CommonSpecies)
    
    def register(self, name: str, calculation_class: Type[ForestCalculation]):
        """
        Register a new calculation type.
        
        Parameters
        ----------
        name : str
            Unique name for the calculation
        calculation_class : Type[ForestCalculation]
            Class that implements ForestCalculation
        """
        if not issubclass(calculation_class, ForestCalculation):
            raise ValueError(f"{calculation_class} must be a subclass of ForestCalculation")
        
        if name in self._calculations:
            logger.warning(f"Overwriting existing calculation: {name}")
        
        self._calculations[name] = calculation_class
        logger.debug(f"Registered calculation: {name}")
    
    def unregister(self, name: str):
        """
        Remove a calculation from the registry.
        
        Parameters
        ----------
        name : str
            Name of the calculation to remove
        """
        if name in self._calculations:
            del self._calculations[name]
            logger.debug(f"Unregistered calculation: {name}")
        else:
            logger.warning(f"Calculation not found: {name}")
    
    def get(self, name: str, **kwargs) -> ForestCalculation:
        """
        Get an instance of a calculation.
        
        Parameters
        ----------
        name : str
            Name of the calculation
        **kwargs : dict
            Arguments to pass to the calculation constructor
            
        Returns
        -------
        ForestCalculation
            Instance of the requested calculation
        """
        if name not in self._calculations:
            raise ValueError(f"Unknown calculation: {name}")
        
        calculation_class = self._calculations[name]
        return calculation_class(**kwargs)
    
    def list_calculations(self) -> List[str]:
        """Get list of available calculation names."""
        return sorted(self._calculations.keys())
    
    def get_calculation_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a calculation.
        
        Parameters
        ----------
        name : str
            Name of the calculation
            
        Returns
        -------
        dict or None
            Information about the calculation or None if not found
        """
        if name not in self._calculations:
            return None
        
        # Create a temporary instance to get metadata
        calc = self.get(name)
        return calc.get_metadata()
    
    def get_all_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all registered calculations."""
        info = {}
        for name in self.list_calculations():
            info[name] = self.get_calculation_info(name)
        return info
    
    def create_calculation_group(self, calculations: List[Dict[str, Any]]) -> List[ForestCalculation]:
        """
        Create multiple calculation instances from configuration.
        
        Parameters
        ----------
        calculations : List[Dict[str, Any]]
            List of calculation configurations with 'name' and optional parameters
            
        Returns
        -------
        List[ForestCalculation]
            List of calculation instances
        """
        instances = []
        for calc_config in calculations:
            if 'name' not in calc_config:
                logger.warning("Skipping calculation without name")
                continue
            
            name = calc_config['name']
            params = {k: v for k, v in calc_config.items() if k != 'name'}
            
            try:
                calc = self.get(name, **params)
                instances.append(calc)
            except Exception as e:
                logger.error(f"Failed to create calculation {name}: {e}")
        
        return instances


# Global registry instance
registry = CalculationRegistry()


# Convenience functions
def register_calculation(name: str, calculation_class: Type[ForestCalculation]):
    """Register a custom calculation with the global registry."""
    registry.register(name, calculation_class)


def get_calculation(name: str, **kwargs) -> ForestCalculation:
    """Get a calculation instance from the global registry."""
    return registry.get(name, **kwargs)


def list_calculations() -> List[str]:
    """List all available calculations in the global registry."""
    return registry.list_calculations()