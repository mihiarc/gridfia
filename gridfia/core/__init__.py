"""
Core functionality for BigMap forest analysis.

This module is organized into submodules:
- calculations: Flexible calculation framework for forest metrics
- processors: High-level data processors and pipelines
- analysis: Analysis and reporting functions
"""

# Import from submodules for backward compatibility
from .calculations import (
    ForestCalculation,
    # Diversity calculations
    SpeciesRichness,
    ShannonDiversity,
    SimpsonDiversity,
    Evenness,
    # Biomass calculations
    TotalBiomass,
    TotalBiomassComparison,
    SpeciesProportion,
    SpeciesPercentage,
    SpeciesGroupProportion,
    BiomassThreshold,
    # Species calculations
    DominantSpecies,
    SpeciesPresence,
    SpeciesDominance,
    RareSpecies,
    CommonSpecies,
    # Registry
    CalculationRegistry,
    registry,
    register_calculation,
    get_calculation,
    list_calculations,
)



__all__ = [
    # Calculations
    'ForestCalculation',
    'SpeciesRichness',
    'ShannonDiversity',
    'SimpsonDiversity',
    'Evenness',
    'TotalBiomass',
    'TotalBiomassComparison',
    'SpeciesProportion',
    'SpeciesPercentage',
    'SpeciesGroupProportion',
    'BiomassThreshold',
    'DominantSpecies',
    'SpeciesPresence',
    'SpeciesDominance',
    'RareSpecies',
    'CommonSpecies',
    'CalculationRegistry',
    'registry',
    'register_calculation',
    'get_calculation',
    'list_calculations',
]