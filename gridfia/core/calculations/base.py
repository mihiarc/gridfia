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
from typing import Dict, Any, Optional, TYPE_CHECKING
import numpy as np
import logging

if TYPE_CHECKING:
    from gridfia.core.analysis.statistical_analysis import StatisticalResult

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

    def calculate_with_stats(
        self,
        biomass_data: np.ndarray,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        seed: Optional[int] = None,
        **kwargs
    ) -> "StatisticalResult":
        """
        Calculate metric with bootstrap confidence intervals.

        This method provides statistical context for the calculation,
        including confidence intervals and standard error estimates.

        Parameters
        ----------
        biomass_data : np.ndarray
            3D array (species, height, width) of biomass values.
        n_bootstrap : int, default=1000
            Number of bootstrap resamples for confidence interval.
        confidence_level : float, default=0.95
            Confidence level for the interval (e.g., 0.95 for 95% CI).
        seed : int, optional
            Random seed for reproducibility.
        **kwargs : dict
            Additional parameters passed to calculate().

        Returns
        -------
        StatisticalResult
            Result containing point estimate, confidence interval,
            standard error, and metadata.

        Examples
        --------
        >>> calc = ShannonDiversityCalculation()
        >>> result = calc.calculate_with_stats(biomass_data, n_bootstrap=500)
        >>> print(f"Shannon: {result.value:.3f} (95% CI: {result.confidence_interval})")

        Notes
        -----
        For pixel-level calculations, this method computes a single
        aggregate statistic (e.g., mean diversity across all pixels)
        and provides confidence intervals for that aggregate.
        """
        from gridfia.core.analysis.statistical_analysis import (
            bootstrap_confidence_interval,
            StatisticalResult,
        )

        # Validate data first
        if not self.validate_data(biomass_data):
            return StatisticalResult(
                value=np.nan,
                confidence_interval=(np.nan, np.nan),
                standard_error=np.nan,
                n_samples=0,
                confidence_level=confidence_level,
                method='bootstrap',
            )

        # Define the statistic function that computes aggregate metric
        def compute_aggregate(data: np.ndarray) -> float:
            """Compute aggregate statistic from calculation result."""
            try:
                # Reshape if flattened
                if data.ndim == 1 and biomass_data.ndim == 3:
                    # Can't easily reshape bootstrap samples for 3D data
                    # Use the flattened data directly
                    result = self.calculate(biomass_data, **kwargs)
                else:
                    result = self.calculate(data, **kwargs)

                # Get aggregate value (mean of non-NaN values)
                if isinstance(result, np.ndarray):
                    valid_values = result[np.isfinite(result)]
                    if len(valid_values) > 0:
                        return float(np.mean(valid_values))
                    return np.nan
                return float(result)
            except Exception:
                return np.nan

        # For 3D biomass data, we bootstrap over the spatial pixels
        # First compute the actual result to get pixel values
        full_result = self.calculate(biomass_data, **kwargs)

        if isinstance(full_result, np.ndarray):
            # Get valid (non-NaN) pixel values for bootstrapping
            valid_pixels = full_result[np.isfinite(full_result)]

            if len(valid_pixels) == 0:
                return StatisticalResult(
                    value=np.nan,
                    confidence_interval=(np.nan, np.nan),
                    standard_error=np.nan,
                    n_samples=0,
                    confidence_level=confidence_level,
                    method='bootstrap',
                )

            # Bootstrap over the pixel values
            return bootstrap_confidence_interval(
                data=valid_pixels,
                statistic_func=np.mean,
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                seed=seed,
            )
        else:
            # Scalar result - can't bootstrap
            return StatisticalResult(
                value=float(full_result),
                confidence_interval=(float(full_result), float(full_result)),
                standard_error=0.0,
                n_samples=1,
                confidence_level=confidence_level,
                method='point_estimate',
            )