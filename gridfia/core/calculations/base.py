"""
Base classes for forest calculations.

This module provides the abstract base class and common functionality
for all forest metric calculations.

NaN Convention for Failed Calculations
--------------------------------------
When calculations fail (due to validation errors, exceptions, or invalid data),
the ForestMetricsProcessor returns NaN values for floating-point output types.
This is intentional to distinguish between:

1. Actual zero values (e.g., zero biomass in a pixel, zero diversity)
2. Failed calculations (e.g., invalid data, processing errors)

For integer output types (e.g., species_richness with uint8):
- Signed integers: -1 is used as a sentinel value (impossible for counts)
- Unsigned integers: max value (e.g., 255 for uint8) is used with a warning

Downstream code should:
- Use np.isnan() to detect failed calculations in float results
- Use np.nansum(), np.nanmean(), etc. for statistics that ignore NaN values
- Check for sentinel values (-1 or max) in integer results
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
            2D array of calculated metric values. For areas with no valid data
            (e.g., no forest), implementations should return appropriate values:
            - Zero for counts/indices where absence is meaningful
            - NaN for ratios/proportions where division would be undefined

        Notes
        -----
        If the calculation fails entirely (e.g., due to invalid input data),
        the ForestMetricsProcessor will return NaN-filled arrays for float types
        or sentinel values for integer types. Individual pixels with valid but
        zero-value results (e.g., no species present) should still return 0.
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