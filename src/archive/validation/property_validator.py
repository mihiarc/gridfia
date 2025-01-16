"""
Property data validation functionality.
"""

import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.validation import explain_validity

from src.utils.error_handler import ErrorHandler, ValidationError

class PropertyValidator:
    """Validates property data for processing."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the property validator.
        
        Args:
            logger: Optional logger instance. If None, creates a new one.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.error_handler = ErrorHandler(self.logger)
        
    def validate_geometries(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Validate and clean property geometries.
        
        Args:
            gdf: Input GeoDataFrame
            
        Returns:
            GeoDataFrame with valid geometries
            
        Raises:
            ValidationError: If there are invalid geometries that cannot be fixed
        """
        try:
            # Check for invalid geometries
            invalid_mask = ~gdf.geometry.is_valid
            invalid_count = invalid_mask.sum()
            
            if invalid_count > 0:
                self.logger.warning(f"Found {invalid_count} invalid geometries")
                
                # Log details of invalid geometries
                for idx, geom in gdf[invalid_mask].geometry.items():
                    self.logger.debug(f"Invalid geometry at index {idx}: {explain_validity(geom)}")
                
                # Attempt to fix invalid geometries
                gdf.loc[invalid_mask, 'geometry'] = gdf[invalid_mask].geometry.buffer(0)
                
                # Check if any geometries are still invalid
                still_invalid = ~gdf.geometry.is_valid
                if still_invalid.any():
                    raise ValidationError(
                        message=f"Unable to fix {still_invalid.sum()} geometries",
                        data_type="property",
                        details={"invalid_indices": still_invalid[still_invalid].index.tolist()}
                    )
                
                self.logger.info("Successfully fixed all invalid geometries")
            
            return gdf
            
        except Exception as e:
            if not isinstance(e, ValidationError):
                e = self.error_handler.handle_validation_error(
                    error=e,
                    data_type="property",
                    details={"stage": "geometry validation"}
                )
            raise e
    
    def validate_required_fields(self, 
                               gdf: gpd.GeoDataFrame,
                               required_fields: List[str]) -> None:
        """
        Validate that required fields are present in the data.
        
        Args:
            gdf: Input GeoDataFrame
            required_fields: List of required field names
            
        Raises:
            ValidationError: If required fields are missing
        """
        try:
            # Convert column names to lowercase for case-insensitive comparison
            columns = set(col.lower() for col in gdf.columns)
            required = set(field.lower() for field in required_fields)
            
            missing_fields = required - columns
            if missing_fields:
                raise ValidationError(
                    message=f"Missing required fields: {', '.join(missing_fields)}",
                    data_type="property",
                    details={"missing_fields": list(missing_fields)}
                )
                
        except Exception as e:
            if not isinstance(e, ValidationError):
                e = self.error_handler.handle_validation_error(
                    error=e,
                    data_type="property",
                    details={"stage": "required fields validation"}
                )
            raise e
    
    def validate_data_types(self, 
                          gdf: gpd.GeoDataFrame,
                          expected_types: Dict[str, str]) -> None:
        """
        Validate data types of specified columns.
        
        Args:
            gdf: Input GeoDataFrame
            expected_types: Dictionary mapping column names to expected data types
            
        Raises:
            ValidationError: If data types don't match expectations
        """
        try:
            type_errors = []
            
            for col, expected_type in expected_types.items():
                if col not in gdf.columns:
                    continue
                    
                actual_type = gdf[col].dtype.name
                if actual_type != expected_type:
                    type_errors.append({
                        "column": col,
                        "expected_type": expected_type,
                        "actual_type": actual_type
                    })
            
            if type_errors:
                raise ValidationError(
                    message="Data type validation failed",
                    data_type="property",
                    details={"type_errors": type_errors}
                )
                
        except Exception as e:
            if not isinstance(e, ValidationError):
                e = self.error_handler.handle_validation_error(
                    error=e,
                    data_type="property",
                    details={"stage": "data type validation"}
                )
            raise e 