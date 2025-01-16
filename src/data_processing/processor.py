import pandas as pd
import geopandas as gpd
from pathlib import Path
import logging
from typing import Dict, Any

from ..config import Config

class DataProcessor:
    """Handles core data processing operations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up basic logging."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def load_property_data(self) -> gpd.GeoDataFrame:
        """Load and validate property data."""
        self.logger.info("Loading property data...")
        
        # Load the main property file (assuming it's a shapefile or geojson)
        property_file = next(self.config.input_dir.glob("*.shp"))
        gdf = gpd.read_file(property_file)
        
        # Basic validation
        missing_fields = [f for f in self.config.required_fields if f not in gdf.columns]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
            
        return gdf
    
    def process_properties(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Process property data."""
        self.logger.info("Processing property data...")
        
        # Basic processing steps
        processed = gdf.copy()
        
        # Add area calculation if not present
        if 'area' not in processed.columns:
            processed['area'] = processed.geometry.area
            
        # Add basic calculations
        processed['centroid_x'] = processed.geometry.centroid.x
        processed['centroid_y'] = processed.geometry.centroid.y
        
        return processed
    
    def save_results(self, gdf: gpd.GeoDataFrame, filename: str = "processed_properties.gpkg"):
        """Save processed results."""
        self.logger.info("Saving results...")
        
        # Create output directory if it doesn't exist
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to GeoPackage format
        output_file = self.config.output_dir / filename
        gdf.to_file(output_file, driver="GPKG")
        
        self.logger.info(f"Results saved to {output_file}")
    
    def run(self) -> Dict[str, Any]:
        """Run the complete processing pipeline."""
        try:
            # Load data
            properties = self.load_property_data()
            
            # Process data
            processed = self.process_properties(properties)
            
            # Save results
            self.save_results(processed)
            
            return {
                "status": "success",
                "properties_processed": len(processed),
                "output_file": str(self.config.output_dir / "processed_properties.gpkg")
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            } 