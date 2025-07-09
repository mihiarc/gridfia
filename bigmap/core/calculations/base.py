"""
Base classes for forest calculations.

This module provides the abstract base class and common functionality
for all forest metric calculations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ForestCalculation(ABC):
    """Abstract base class for forest calculations."""
    
    def __init__(self, name: str, description: str, units: str, **kwargs):
        """
        Initialize a forest calculation.
        
        Parameters
        ----------
        name : str
            Unique name for the calculation
        description : str
            Human-readable description
        units : str
            Units of the calculated metric
        **kwargs : dict
            Additional configuration parameters
        """
        self.name = name
        self.description = description
        self.units = units
        self.config = kwargs
    
    @abstractmethod
    def calculate(self, biomass_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Calculate metric from biomass data.
        
        Parameters
        ----------
        biomass_data : np.ndarray
            3D array (species, height, width) of biomass values
        **kwargs : dict
            Additional calculation parameters
            
        Returns
        -------
        np.ndarray
            2D array of calculated metric values
        """
        pass
    
    @abstractmethod
    def validate_data(self, biomass_data: np.ndarray) -> bool:
        """
        Validate input data for this calculation.
        
        Parameters
        ----------
        biomass_data : np.ndarray
            Input biomass data to validate
            
        Returns
        -------
        bool
            True if data is valid, False otherwise
        """
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata for this calculation."""
        return {
            'name': self.name,
            'description': self.description,
            'units': self.units,
            'config': self.config,
            'dtype': self.get_output_dtype()
        }
    
    def get_output_dtype(self) -> np.dtype:
        """Get appropriate numpy dtype for output."""
        return np.float32
    
    def preprocess_data(self, biomass_data: np.ndarray) -> np.ndarray:
        """
        Preprocess data before calculation.
        
        Can be overridden by subclasses for custom preprocessing.
        """
        return biomass_data
    
    def postprocess_result(self, result: np.ndarray) -> np.ndarray:
        """
        Postprocess calculation result.
        
        Can be overridden by subclasses for custom postprocessing.
        """
        return result