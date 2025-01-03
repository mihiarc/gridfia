"""
Data validation module for geospatial data.
"""
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional

import geopandas as gpd
from shapely.geometry import Polygon

class DataValidator:
    """Validates geospatial data before processing."""
    
    def __init__(
        self,
        required_fields: List[str],
        geometry_type: str = "Polygon",
        srid: int = 2264,
        custom_rules: Optional[Dict[str, callable]] = None
    ):
        """Initialize validator with validation rules.
        
        Args:
            required_fields: List of field names that must be present
            geometry_type: Expected geometry type (default: Polygon)
            srid: Expected spatial reference ID (default: 2264)
            custom_rules: Optional dict of field names to validation functions
        """
        self.required_fields = required_fields
        self.geometry_type = geometry_type
        self.srid = srid
        self.custom_rules = custom_rules or {}
    
    def validate_geodataframe(
        self,
        gdf: gpd.GeoDataFrame,
        dataset_name: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate a GeoDataFrame against all rules.
        
        Args:
            gdf: GeoDataFrame to validate
            dataset_name: Name of the dataset for reporting
            
        Returns:
            Tuple of (success, report)
        """
        issues = []
        
        # Check required fields
        missing_fields = [
            field for field in self.required_fields 
            if field not in gdf.columns
        ]
        if missing_fields:
            issues.append({
                "type": "missing_fields",
                "fields": missing_fields,
                "count": len(missing_fields)
            })
        
        # Check CRS
        if gdf.crs is None:
            issues.append({
                "type": "missing_crs",
                "details": "No CRS defined"
            })
        elif gdf.crs.to_epsg() != self.srid:
            issues.append({
                "type": "wrong_crs",
                "expected": self.srid,
                "found": gdf.crs.to_epsg()
            })
        
        # Check geometry type
        invalid_geom_types = gdf.geometry.apply(
            lambda x: x is not None and not isinstance(x, eval(self.geometry_type))
        )
        if invalid_geom_types.any():
            issues.append({
                "type": "invalid_geometry_type",
                "expected": self.geometry_type,
                "count": invalid_geom_types.sum()
            })
        
        # Check for null geometries
        null_geoms = gdf.geometry.isna()
        if null_geoms.any():
            issues.append({
                "type": "null_geometries",
                "count": null_geoms.sum()
            })
        
        # Check geometry validity
        invalid_geoms = ~gdf.geometry.is_valid
        if invalid_geoms.any():
            issues.append({
                "type": "invalid_geometries",
                "count": invalid_geoms.sum(),
                "details": "Self-intersecting or invalid polygons"
            })
        
        # Apply custom validation rules
        for field, rule in self.custom_rules.items():
            if field in gdf.columns:
                invalid = ~gdf[field].apply(rule)
                if invalid.any():
                    issues.append({
                        "type": "custom_validation_error",
                        "field": field,
                        "count": invalid.sum()
                    })
        
        # Create validation report
        report = {
            "dataset": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "record_count": len(gdf),
            "issues": issues
        }
        
        return len(issues) == 0, report
    
    def validate_file(
        self,
        file_path: str,
        dataset_name: Optional[str] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate a geospatial file.
        
        Args:
            file_path: Path to the file to validate
            dataset_name: Optional name for the dataset
            
        Returns:
            Tuple of (success, report)
        """
        try:
            if file_path.endswith('.parquet'):
                gdf = gpd.read_parquet(file_path)
            else:
                gdf = gpd.read_file(file_path)
            
            return self.validate_geodataframe(
                gdf,
                dataset_name or file_path
            )
        except Exception as e:
            return False, {
                "dataset": dataset_name or file_path,
                "timestamp": datetime.now().isoformat(),
                "record_count": 0,
                "issues": [{
                    "type": "validation_error",
                    "details": str(e)
                }]
            } 