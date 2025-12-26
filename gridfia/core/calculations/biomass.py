"""
Biomass calculations for forest analysis.

This module provides calculations for various biomass metrics including
total biomass, biomass proportions, and biomass comparisons.
"""

import numpy as np
import logging
from typing import Optional, List, Dict, Any

from .base import ForestCalculation

logger = logging.getLogger(__name__)


class TotalBiomass(ForestCalculation):
    """Calculate total biomass across species."""
    
    def __init__(self, exclude_total_layer: bool = True, **kwargs):
        """
        Initialize total biomass calculation.
        
        Parameters
        ----------
        exclude_total_layer : bool
            Whether to exclude first layer (pre-calculated total)
        """
        super().__init__(
            name="total_biomass",
            description="Total above-ground biomass across species",
            units="Mg/ha",
            exclude_total_layer=exclude_total_layer,
            **kwargs
        )
    
    def calculate(self, biomass_data: np.ndarray, **kwargs) -> np.ndarray:
        """Sum biomass across individual species layers."""
        exclude_total = kwargs.get('exclude_total_layer', self.config['exclude_total_layer'])
        
        if exclude_total and biomass_data.shape[0] > 1:
            # Sum only individual species layers (exclude pre-calculated total)
            return np.sum(biomass_data[1:], axis=0)
        else:
            # Sum all layers or use single layer
            if biomass_data.shape[0] == 1:
                return biomass_data[0]
            return np.sum(biomass_data, axis=0)
    
    def validate_data(self, biomass_data: np.ndarray) -> bool:
        return biomass_data.ndim == 3 and biomass_data.shape[0] > 0


class TotalBiomassComparison(ForestCalculation):
    """Compare calculated total biomass with pre-calculated total layer."""
    
    def __init__(self, tolerance: float = 0.01, **kwargs):
        """
        Initialize biomass comparison calculation.
        
        Parameters
        ----------
        tolerance : float
            Tolerance for difference (fraction of total)
        """
        super().__init__(
            name="total_biomass_comparison",
            description="Difference between calculated and pre-calculated total biomass",
            units="Mg/ha",
            tolerance=tolerance,
            **kwargs
        )
    
    def calculate(self, biomass_data: np.ndarray, **kwargs) -> np.ndarray:
        """Calculate absolute difference between totals."""
        if biomass_data.shape[0] <= 1:
            logger.warning("Cannot compare totals with only one layer - returning NaN")
            # Return NaN instead of zeros to indicate calculation failure
            # This distinguishes from actual zero difference between totals
            return np.full(biomass_data.shape[1:], np.nan, dtype=np.float32)
        
        pre_calculated_total = biomass_data[0]
        calculated_total = np.sum(biomass_data[1:], axis=0)
        
        return np.abs(pre_calculated_total - calculated_total)
    
    def validate_data(self, biomass_data: np.ndarray) -> bool:
        return biomass_data.ndim == 3 and biomass_data.shape[0] > 1


class SpeciesProportion(ForestCalculation):
    """Calculate proportion of biomass for a specific species."""
    
    def __init__(self, species_index: int, species_name: Optional[str] = None, **kwargs):
        """
        Initialize species proportion calculation.
        
        Parameters
        ----------
        species_index : int
            Index of species in the biomass array
        species_name : str, optional
            Name of the species for documentation
        """
        name = f"species_{species_index}_proportion"
        if species_name:
            description = f"Proportion of biomass from {species_name}"
        else:
            description = f"Proportion of biomass from species index {species_index}"
            
        super().__init__(
            name=name,
            description=description,
            units="fraction",
            species_index=species_index,
            species_name=species_name,
            **kwargs
        )
    
    def calculate(self, biomass_data: np.ndarray, **kwargs) -> np.ndarray:
        """Calculate proportion of total biomass from specified species."""
        species_index = kwargs.get('species_index', self.config['species_index'])
        
        if species_index >= biomass_data.shape[0]:
            raise ValueError(f"Species index {species_index} out of range")
        
        # Get species biomass
        species_biomass = biomass_data[species_index]
        
        # Calculate total biomass (excluding pre-calculated if index 0)
        if species_index == 0:
            raise ValueError("Cannot calculate proportion for total layer (index 0)")
        
        total_biomass = np.sum(biomass_data[1:], axis=0)
        
        # Calculate proportion
        proportion = np.zeros_like(species_biomass)
        mask = total_biomass > 0
        proportion[mask] = species_biomass[mask] / total_biomass[mask]
        
        return proportion
    
    def validate_data(self, biomass_data: np.ndarray) -> bool:
        return (biomass_data.ndim == 3 and 
                biomass_data.shape[0] > self.config['species_index'])


class SpeciesPercentage(SpeciesProportion):
    """Calculate percentage of biomass for a specific species."""
    
    def __init__(self, species_index: int, species_name: Optional[str] = None, **kwargs):
        """
        Initialize species percentage calculation.
        
        Parameters
        ----------
        species_index : int
            Index of species in the biomass array
        species_name : str, optional
            Name of the species for documentation
        """
        super().__init__(species_index, species_name, **kwargs)
        self.name = f"species_{species_index}_percentage"
        self.units = "percent"
        if species_name:
            self.description = f"Percentage of biomass from {species_name}"
        else:
            self.description = f"Percentage of biomass from species index {species_index}"
    
    def calculate(self, biomass_data: np.ndarray, **kwargs) -> np.ndarray:
        """Calculate percentage by multiplying proportion by 100."""
        proportion = super().calculate(biomass_data, **kwargs)
        return proportion * 100.0


class SpeciesGroupProportion(ForestCalculation):
    """Calculate combined proportion of biomass from multiple species."""
    
    def __init__(self, species_indices: List[int], group_name: str, **kwargs):
        """
        Initialize species group proportion calculation.
        
        Parameters
        ----------
        species_indices : List[int]
            Indices of species to combine
        group_name : str
            Name of the species group
        """
        super().__init__(
            name=f"{group_name.lower().replace(' ', '_')}_proportion",
            description=f"Combined proportion of biomass from {group_name}",
            units="fraction",
            species_indices=species_indices,
            group_name=group_name,
            **kwargs
        )
    
    def calculate(self, biomass_data: np.ndarray, **kwargs) -> np.ndarray:
        """Calculate combined proportion from species group."""
        species_indices = kwargs.get('species_indices', self.config['species_indices'])
        
        # Validate indices
        for idx in species_indices:
            if idx >= biomass_data.shape[0] or idx == 0:
                raise ValueError(f"Invalid species index: {idx}")
        
        # Sum biomass from species group
        group_biomass = np.sum(biomass_data[species_indices], axis=0)
        
        # Calculate total biomass (excluding pre-calculated)
        total_biomass = np.sum(biomass_data[1:], axis=0)
        
        # Calculate proportion
        proportion = np.zeros_like(group_biomass)
        mask = total_biomass > 0
        proportion[mask] = group_biomass[mask] / total_biomass[mask]
        
        return proportion
    
    def validate_data(self, biomass_data: np.ndarray) -> bool:
        if biomass_data.ndim != 3:
            return False
        
        # Check all indices are valid
        for idx in self.config['species_indices']:
            if idx >= biomass_data.shape[0]:
                return False
        
        return True


class BiomassThreshold(ForestCalculation):
    """Identify areas above/below biomass threshold."""
    
    def __init__(self, threshold: float, above: bool = True, **kwargs):
        """
        Initialize biomass threshold calculation.
        
        Parameters
        ----------
        threshold : float
            Biomass threshold value (Mg/ha)
        above : bool
            If True, identify areas above threshold; if False, below
        """
        direction = "above" if above else "below"
        super().__init__(
            name=f"biomass_{direction}_{threshold}",
            description=f"Areas with biomass {direction} {threshold} Mg/ha",
            units="boolean",
            threshold=threshold,
            above=above,
            **kwargs
        )
    
    def calculate(self, biomass_data: np.ndarray, **kwargs) -> np.ndarray:
        """Identify areas meeting threshold criteria."""
        threshold = kwargs.get('threshold', self.config['threshold'])
        above = kwargs.get('above', self.config['above'])
        
        # Calculate total biomass
        total_calc = TotalBiomass()
        total_biomass = total_calc.calculate(biomass_data)
        
        # Apply threshold
        if above:
            return (total_biomass > threshold).astype(np.uint8)
        else:
            return (total_biomass <= threshold).astype(np.uint8)
    
    def validate_data(self, biomass_data: np.ndarray) -> bool:
        return biomass_data.ndim == 3 and biomass_data.shape[0] > 0
    
    def get_output_dtype(self) -> np.dtype:
        return np.uint8