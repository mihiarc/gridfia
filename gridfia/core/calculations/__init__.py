"""
Forest calculations submodule.

This module provides a flexible framework for calculating various forest
metrics from multi-species biomass data.
"""

# Import base class
from .base import ForestCalculation

# Import all calculation types
from .diversity import (
    SpeciesRichness,
    ShannonDiversity,
    SimpsonDiversity,
    Evenness
)

from .biomass import (
    TotalBiomass,
    TotalBiomassComparison,
    SpeciesProportion,
    SpeciesPercentage,
    SpeciesGroupProportion,
    BiomassThreshold
)

from .species import (
    DominantSpecies,
    SpeciesPresence,
    SpeciesDominance,
    RareSpecies,
    CommonSpecies
)

# Import registry and convenience functions
from .registry import (
    CalculationRegistry,
    registry,
    register_calculation,
    get_calculation,
    list_calculations
)

__all__ = [
    # Base class
    'ForestCalculation',
    
    # Diversity calculations
    'SpeciesRichness',
    'ShannonDiversity',
    'SimpsonDiversity',
    'Evenness',
    
    # Biomass calculations
    'TotalBiomass',
    'TotalBiomassComparison',
    'SpeciesProportion',
    'SpeciesPercentage',
    'SpeciesGroupProportion',
    'BiomassThreshold',
    
    # Species calculations
    'DominantSpecies',
    'SpeciesPresence',
    'SpeciesDominance',
    'RareSpecies',
    'CommonSpecies',
    
    # Registry
    'CalculationRegistry',
    'registry',
    'register_calculation',
    'get_calculation',
    'list_calculations',
]