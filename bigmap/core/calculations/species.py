"""
Species-specific calculations for forest analysis.

This module provides calculations focused on individual species metrics
such as dominant species identification and species-specific analyses.
"""

import numpy as np
import logging
from typing import Optional, List, Dict, Any

from .base import ForestCalculation

logger = logging.getLogger(__name__)


class DominantSpecies(ForestCalculation):
    """Identify the dominant species by biomass at each pixel."""
    
    def __init__(self, exclude_total_layer: bool = True, min_biomass: float = 0.0, **kwargs):
        """
        Initialize dominant species calculation.
        
        Parameters
        ----------
        exclude_total_layer : bool
            Whether to exclude first layer (pre-calculated total)
        min_biomass : float
            Minimum biomass threshold to consider
        """
        super().__init__(
            name="dominant_species",
            description="Index of species with maximum biomass",
            units="species_index",
            exclude_total_layer=exclude_total_layer,
            min_biomass=min_biomass,
            **kwargs
        )
    
    def calculate(self, biomass_data: np.ndarray, **kwargs) -> np.ndarray:
        """Find species with maximum biomass at each pixel."""
        exclude_total = kwargs.get('exclude_total_layer', self.config['exclude_total_layer'])
        min_biomass = kwargs.get('min_biomass', self.config['min_biomass'])
        
        # Select appropriate data
        if exclude_total and biomass_data.shape[0] > 1:
            species_data = biomass_data[1:]
            # Adjust indices to account for excluded layer
            index_offset = 1
        else:
            species_data = biomass_data
            index_offset = 0
        
        # Find maximum biomass and corresponding species
        max_biomass = np.max(species_data, axis=0)
        dominant = np.argmax(species_data, axis=0)
        
        # Apply minimum biomass threshold
        mask = max_biomass > min_biomass
        result = np.zeros(dominant.shape, dtype=np.uint8)
        result[mask] = dominant[mask] + index_offset
        
        return result
    
    def validate_data(self, biomass_data: np.ndarray) -> bool:
        return biomass_data.ndim == 3 and biomass_data.shape[0] > 0
    
    def get_output_dtype(self) -> np.dtype:
        return np.uint8


class SpeciesPresence(ForestCalculation):
    """Determine presence/absence of a specific species."""
    
    def __init__(self, species_index: int, species_name: Optional[str] = None, 
                 threshold: float = 0.0, **kwargs):
        """
        Initialize species presence calculation.
        
        Parameters
        ----------
        species_index : int
            Index of species in the biomass array
        species_name : str, optional
            Name of the species for documentation
        threshold : float
            Minimum biomass to consider species present
        """
        name = f"species_{species_index}_presence"
        if species_name:
            description = f"Presence of {species_name}"
        else:
            description = f"Presence of species index {species_index}"
            
        super().__init__(
            name=name,
            description=description,
            units="boolean",
            species_index=species_index,
            species_name=species_name,
            threshold=threshold,
            **kwargs
        )
    
    def calculate(self, biomass_data: np.ndarray, **kwargs) -> np.ndarray:
        """Determine species presence based on threshold."""
        species_index = kwargs.get('species_index', self.config['species_index'])
        threshold = kwargs.get('threshold', self.config['threshold'])
        
        if species_index >= biomass_data.shape[0]:
            raise ValueError(f"Species index {species_index} out of range")
        
        species_biomass = biomass_data[species_index]
        return (species_biomass > threshold).astype(np.uint8)
    
    def validate_data(self, biomass_data: np.ndarray) -> bool:
        return (biomass_data.ndim == 3 and 
                biomass_data.shape[0] > self.config['species_index'])
    
    def get_output_dtype(self) -> np.dtype:
        return np.uint8


class SpeciesDominance(ForestCalculation):
    """Calculate dominance percentage for a specific species."""
    
    def __init__(self, species_index: int, species_name: Optional[str] = None, **kwargs):
        """
        Initialize species dominance calculation.
        
        Parameters
        ----------
        species_index : int
            Index of species in the biomass array
        species_name : str, optional
            Name of the species for documentation
        """
        name = f"species_{species_index}_dominance"
        if species_name:
            description = f"Dominance percentage of {species_name}"
        else:
            description = f"Dominance percentage of species index {species_index}"
            
        super().__init__(
            name=name,
            description=description,
            units="percent",
            species_index=species_index,
            species_name=species_name,
            **kwargs
        )
    
    def calculate(self, biomass_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Calculate percentage of pixels where species is dominant.
        
        Returns a single value expanded to match spatial dimensions.
        """
        species_index = kwargs.get('species_index', self.config['species_index'])
        
        if species_index >= biomass_data.shape[0] or species_index == 0:
            raise ValueError(f"Invalid species index: {species_index}")
        
        # Find dominant species at each pixel
        dominant_calc = DominantSpecies()
        dominant = dominant_calc.calculate(biomass_data)
        
        # Calculate percentage where this species is dominant
        total_pixels = dominant.size
        dominant_pixels = np.sum(dominant == species_index)
        
        dominance_percent = (dominant_pixels / total_pixels) * 100.0
        
        # Return as array matching spatial dimensions
        result = np.full(dominant.shape, dominance_percent, dtype=np.float32)
        return result
    
    def validate_data(self, biomass_data: np.ndarray) -> bool:
        return (biomass_data.ndim == 3 and 
                biomass_data.shape[0] > self.config['species_index'])


class RareSpecies(ForestCalculation):
    """Identify rare species based on occurrence threshold."""
    
    def __init__(self, occurrence_threshold: float = 0.01, 
                 biomass_threshold: float = 0.0, **kwargs):
        """
        Initialize rare species calculation.
        
        Parameters
        ----------
        occurrence_threshold : float
            Maximum fraction of pixels for species to be considered rare
        biomass_threshold : float
            Minimum biomass to count species as present
        """
        super().__init__(
            name="rare_species",
            description="Count of rare species per pixel",
            units="count",
            occurrence_threshold=occurrence_threshold,
            biomass_threshold=biomass_threshold,
            **kwargs
        )
    
    def calculate(self, biomass_data: np.ndarray, **kwargs) -> np.ndarray:
        """Count rare species at each pixel."""
        occurrence_threshold = kwargs.get('occurrence_threshold', 
                                        self.config['occurrence_threshold'])
        biomass_threshold = kwargs.get('biomass_threshold', 
                                     self.config['biomass_threshold'])
        
        # Skip total layer
        species_data = biomass_data[1:] if biomass_data.shape[0] > 1 else biomass_data
        
        n_species, height, width = species_data.shape
        total_pixels = height * width
        
        # Calculate occurrence frequency for each species
        occurrence_freq = np.zeros(n_species)
        for i in range(n_species):
            occurrence_freq[i] = np.sum(species_data[i] > biomass_threshold) / total_pixels
        
        # Identify rare species
        rare_species_mask = occurrence_freq < occurrence_threshold
        
        # Count rare species at each pixel
        rare_count = np.zeros((height, width), dtype=np.uint8)
        for i in range(n_species):
            if rare_species_mask[i]:
                rare_count += (species_data[i] > biomass_threshold).astype(np.uint8)
        
        return rare_count
    
    def validate_data(self, biomass_data: np.ndarray) -> bool:
        return biomass_data.ndim == 3 and biomass_data.shape[0] > 0
    
    def get_output_dtype(self) -> np.dtype:
        return np.uint8


class CommonSpecies(ForestCalculation):
    """Identify common species based on occurrence threshold."""
    
    def __init__(self, occurrence_threshold: float = 0.10, 
                 biomass_threshold: float = 0.0, **kwargs):
        """
        Initialize common species calculation.
        
        Parameters
        ----------
        occurrence_threshold : float
            Minimum fraction of pixels for species to be considered common
        biomass_threshold : float
            Minimum biomass to count species as present
        """
        super().__init__(
            name="common_species",
            description="Count of common species per pixel",
            units="count",
            occurrence_threshold=occurrence_threshold,
            biomass_threshold=biomass_threshold,
            **kwargs
        )
    
    def calculate(self, biomass_data: np.ndarray, **kwargs) -> np.ndarray:
        """Count common species at each pixel."""
        occurrence_threshold = kwargs.get('occurrence_threshold', 
                                        self.config['occurrence_threshold'])
        biomass_threshold = kwargs.get('biomass_threshold', 
                                     self.config['biomass_threshold'])
        
        # Skip total layer
        species_data = biomass_data[1:] if biomass_data.shape[0] > 1 else biomass_data
        
        n_species, height, width = species_data.shape
        total_pixels = height * width
        
        # Calculate occurrence frequency for each species
        occurrence_freq = np.zeros(n_species)
        for i in range(n_species):
            occurrence_freq[i] = np.sum(species_data[i] > biomass_threshold) / total_pixels
        
        # Identify common species
        common_species_mask = occurrence_freq >= occurrence_threshold
        
        # Count common species at each pixel
        common_count = np.zeros((height, width), dtype=np.uint8)
        for i in range(n_species):
            if common_species_mask[i]:
                common_count += (species_data[i] > biomass_threshold).astype(np.uint8)
        
        return common_count
    
    def validate_data(self, biomass_data: np.ndarray) -> bool:
        return biomass_data.ndim == 3 and biomass_data.shape[0] > 0
    
    def get_output_dtype(self) -> np.dtype:
        return np.uint8