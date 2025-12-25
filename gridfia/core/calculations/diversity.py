"""
Diversity calculations for forest analysis.

This module provides calculations for various diversity metrics including
species richness, Shannon diversity index, Simpson diversity index, and evenness.
"""

import numpy as np
import logging
from typing import Optional

from .base import ForestCalculation

logger = logging.getLogger(__name__)


class SpeciesRichness(ForestCalculation):
    """Calculate species richness (count of species with biomass > threshold)."""
    
    def __init__(self, biomass_threshold: float = 0.0, exclude_total_layer: bool = True, **kwargs):
        """
        Initialize species richness calculation.
        
        Parameters
        ----------
        biomass_threshold : float
            Minimum biomass to count species as present
        exclude_total_layer : bool
            Whether to exclude first layer (pre-calculated total)
        """
        super().__init__(
            name="species_richness",
            description="Number of tree species per pixel",
            units="count",
            biomass_threshold=biomass_threshold,
            exclude_total_layer=exclude_total_layer,
            **kwargs
        )
    
    def calculate(self, biomass_data: np.ndarray, **kwargs) -> np.ndarray:
        """Count non-zero species per pixel."""
        threshold = kwargs.get('biomass_threshold', self.config['biomass_threshold'])
        exclude_total = kwargs.get('exclude_total_layer', self.config['exclude_total_layer'])
        
        if exclude_total and biomass_data.shape[0] > 1:
            # Exclude first layer (pre-calculated total) and count individual species
            return np.count_nonzero(biomass_data[1:] > threshold, axis=0)
        else:
            # Count all layers
            return np.count_nonzero(biomass_data > threshold, axis=0)
    
    def validate_data(self, biomass_data: np.ndarray) -> bool:
        return biomass_data.ndim == 3 and biomass_data.shape[0] > 0
    
    def get_output_dtype(self) -> np.dtype:
        return np.uint8


class ShannonDiversity(ForestCalculation):
    """Calculate Shannon diversity index."""
    
    def __init__(self, exclude_total_layer: bool = True, base: str = 'e', **kwargs):
        """
        Initialize Shannon diversity calculation.
        
        Parameters
        ----------
        exclude_total_layer : bool
            Whether to exclude first layer (pre-calculated total)
        base : str
            Logarithm base ('e' for natural log, '2' for log2)
        """
        super().__init__(
            name="shannon_diversity",
            description="Shannon diversity index",
            units="index",
            exclude_total_layer=exclude_total_layer,
            base=base,
            **kwargs
        )
    
    def calculate(self, biomass_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Calculate Shannon diversity index.
        
        H' = -Σ(pi * ln(pi)) where pi is proportion of species i
        """
        exclude_total = kwargs.get('exclude_total_layer', self.config['exclude_total_layer'])
        base = kwargs.get('base', self.config['base'])
        
        # Select appropriate data
        if exclude_total and biomass_data.shape[0] > 1:
            species_data = biomass_data[1:]
        else:
            species_data = biomass_data
        
        # Calculate total biomass per pixel
        total_biomass = np.sum(species_data, axis=0)
        
        # Initialize output
        n_species, height, width = species_data.shape
        shannon = np.zeros((height, width), dtype=np.float32)
        
        # Mask for pixels with biomass
        valid_mask = total_biomass > 0
        
        if np.any(valid_mask):
            # Calculate proportions for valid pixels
            proportions = np.zeros_like(species_data, dtype=np.float32)
            proportions[:, valid_mask] = species_data[:, valid_mask] / total_biomass[valid_mask]
            
            # Calculate Shannon index
            if base == '2':
                log_func = np.log2
            else:  # default to natural log
                log_func = np.log

            # Only calculate for non-zero proportions to avoid log(0)
            mask = proportions > 0
            shannon_contrib = np.zeros_like(proportions)
            shannon_contrib[mask] = -proportions[mask] * log_func(proportions[mask])

            # Sum across species
            shannon = np.sum(shannon_contrib, axis=0)
        
        return shannon
    
    def validate_data(self, biomass_data: np.ndarray) -> bool:
        return biomass_data.ndim == 3 and biomass_data.shape[0] > 0


class SimpsonDiversity(ForestCalculation):
    """Calculate Simpson diversity index."""
    
    def __init__(self, exclude_total_layer: bool = True, inverse: bool = True, **kwargs):
        """
        Initialize Simpson diversity calculation.
        
        Parameters
        ----------
        exclude_total_layer : bool
            Whether to exclude first layer (pre-calculated total)
        inverse : bool
            Whether to return inverse Simpson index (1/D)
        """
        super().__init__(
            name="simpson_diversity",
            description="Simpson diversity index",
            units="index",
            exclude_total_layer=exclude_total_layer,
            inverse=inverse,
            **kwargs
        )
    
    def calculate(self, biomass_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Calculate Simpson diversity index.
        
        D = Σ(pi^2) where pi is proportion of species i
        Returns 1/D if inverse=True (inverse Simpson index)
        """
        exclude_total = kwargs.get('exclude_total_layer', self.config['exclude_total_layer'])
        inverse = kwargs.get('inverse', self.config['inverse'])
        
        # Select appropriate data
        if exclude_total and biomass_data.shape[0] > 1:
            species_data = biomass_data[1:]
        else:
            species_data = biomass_data
        
        # Calculate total biomass per pixel
        total_biomass = np.sum(species_data, axis=0)
        
        # Initialize output
        height, width = species_data.shape[1:]
        simpson = np.zeros((height, width), dtype=np.float32)
        
        # Mask for pixels with biomass
        valid_mask = total_biomass > 0
        
        if np.any(valid_mask):
            # Calculate proportions for valid pixels
            proportions = np.zeros_like(species_data, dtype=np.float32)
            proportions[:, valid_mask] = species_data[:, valid_mask] / total_biomass[valid_mask]
            
            # Calculate Simpson index (sum of squared proportions)
            simpson = np.sum(proportions ** 2, axis=0)
            
            # Apply inverse if requested
            if inverse:
                # Avoid division by zero
                mask = simpson > 0
                result = np.ones_like(simpson)
                result[mask] = 1.0 / simpson[mask]
                simpson = result
        
        return simpson
    
    def validate_data(self, biomass_data: np.ndarray) -> bool:
        return biomass_data.ndim == 3 and biomass_data.shape[0] > 0


class Evenness(ForestCalculation):
    """Calculate species evenness (Pielou's evenness)."""
    
    def __init__(self, exclude_total_layer: bool = True, **kwargs):
        """
        Initialize evenness calculation.
        
        Parameters
        ----------
        exclude_total_layer : bool
            Whether to exclude first layer (pre-calculated total)
        """
        super().__init__(
            name="evenness",
            description="Pielou's evenness index",
            units="index",
            exclude_total_layer=exclude_total_layer,
            **kwargs
        )
    
    def calculate(self, biomass_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Calculate Pielou's evenness.
        
        J = H' / H'max = H' / ln(S)
        where H' is Shannon diversity and S is species richness
        """
        exclude_total = kwargs.get('exclude_total_layer', self.config['exclude_total_layer'])
        
        # Calculate Shannon diversity
        shannon_calc = ShannonDiversity(exclude_total_layer=exclude_total)
        shannon = shannon_calc.calculate(biomass_data)
        
        # Calculate species richness
        richness_calc = SpeciesRichness(exclude_total_layer=exclude_total)
        richness = richness_calc.calculate(biomass_data)
        
        # Calculate evenness
        evenness = np.zeros_like(shannon)
        
        # Only calculate where richness > 1 (need at least 2 species for evenness)
        mask = richness > 1
        if np.any(mask):
            # Maximum possible Shannon diversity = ln(richness)
            h_max = np.log(richness[mask])
            evenness[mask] = shannon[mask] / h_max
        
        return evenness
    
    def validate_data(self, biomass_data: np.ndarray) -> bool:
        return biomass_data.ndim == 3 and biomass_data.shape[0] > 0