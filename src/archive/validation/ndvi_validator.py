"""
NDVI data validation functionality.
"""

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

import rasterio
import numpy as np
from rasterio.errors import RasterioError

from src.utils.error_handler import ErrorHandler, ValidationError

class NDVIValidator:
    """Validates NDVI data for processing."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the NDVI validator.
        
        Args:
            logger: Optional logger instance. If None, creates a new one.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.error_handler = ErrorHandler(self.logger)
        
    def validate_ndvi_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Validate an NDVI file and return its metadata.
        
        Args:
            file_path: Path to the NDVI file
            
        Returns:
            Dictionary containing file metadata
            
        Raises:
            ValidationError: If the file is invalid or doesn't meet requirements
        """
        try:
            with rasterio.open(file_path) as src:
                metadata = {
                    'crs': src.crs.to_string(),
                    'bounds': src.bounds,
                    'resolution': src.res,
                    'shape': src.shape,
                    'nodata': src.nodata,
                    'dtype': src.dtypes[0]
                }
                
                # Validate basic requirements
                if not src.crs:
                    raise ValidationError(
                        message="NDVI file has no CRS defined",
                        data_type="ndvi",
                        details={"file": str(file_path)}
                    )
                
                # Check for valid resolution
                if any(res <= 0 for res in src.res):
                    raise ValidationError(
                        message="Invalid pixel resolution",
                        data_type="ndvi",
                        details={
                            "file": str(file_path),
                            "resolution": src.res
                        }
                    )
                
                # Read a small sample to check data range
                window = rasterio.windows.Window(0, 0, 100, 100)
                sample = src.read(1, window=window)
                
                if src.nodata is not None:
                    sample = sample[sample != src.nodata]
                
                if len(sample) > 0:
                    metadata.update({
                        'min_value': float(np.min(sample)),
                        'max_value': float(np.max(sample))
                    })
                    
                    # Check for reasonable NDVI range (allowing for scaled values)
                    max_abs_value = max(abs(metadata['min_value']), 
                                     abs(metadata['max_value']))
                    if max_abs_value > 10000:  # Assuming scaled int16 values
                        raise ValidationError(
                            message="NDVI values out of expected range",
                            data_type="ndvi",
                            details={
                                "file": str(file_path),
                                "min_value": metadata['min_value'],
                                "max_value": metadata['max_value']
                            }
                        )
                
                return metadata
                
        except RasterioError as e:
            raise self.error_handler.handle_validation_error(
                error=e,
                data_type="ndvi",
                details={
                    "file": str(file_path),
                    "stage": "file reading"
                }
            )
        except Exception as e:
            if not isinstance(e, ValidationError):
                e = self.error_handler.handle_validation_error(
                    error=e,
                    data_type="ndvi",
                    details={
                        "file": str(file_path),
                        "stage": "validation"
                    }
                )
            raise e
    
    def validate_temporal_consistency(self, 
                                   metadata_list: List[Dict[str, Any]]) -> None:
        """
        Validate consistency across a temporal series of NDVI files.
        
        Args:
            metadata_list: List of metadata dictionaries from validate_ndvi_file
            
        Raises:
            ValidationError: If files are inconsistent
        """
        if not metadata_list:
            return
            
        try:
            reference = metadata_list[0]
            
            for i, metadata in enumerate(metadata_list[1:], 1):
                # Check CRS consistency
                if metadata['crs'] != reference['crs']:
                    raise ValidationError(
                        message="Inconsistent CRS across NDVI files",
                        data_type="ndvi",
                        details={
                            "reference_crs": reference['crs'],
                            "file_crs": metadata['crs'],
                            "file_index": i
                        }
                    )
                
                # Check resolution consistency
                if metadata['resolution'] != reference['resolution']:
                    raise ValidationError(
                        message="Inconsistent resolution across NDVI files",
                        data_type="ndvi",
                        details={
                            "reference_resolution": reference['resolution'],
                            "file_resolution": metadata['resolution'],
                            "file_index": i
                        }
                    )
                
                # Check for similar bounds (allowing small differences)
                bounds_diff = [abs(a - b) for a, b in 
                             zip(metadata['bounds'], reference['bounds'])]
                if any(diff > 1e-6 for diff in bounds_diff):
                    self.logger.warning(
                        f"File {i} has different bounds: {metadata['bounds']} "
                        f"vs reference: {reference['bounds']}"
                    )
                    
        except Exception as e:
            if not isinstance(e, ValidationError):
                e = self.error_handler.handle_validation_error(
                    error=e,
                    data_type="ndvi",
                    details={"stage": "temporal consistency validation"}
                )
            raise e 