"""
Batch processing functionality for property and NDVI data.
"""

import logging
from typing import Optional, Dict, List, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import geopandas as gpd

from src.utils.error_handler import ErrorHandler, ProcessingError
from src.data_processing.loading.property_loader import PropertyLoader
from src.data_processing.loading.ndvi_loader import NDVILoader
from src.data_processing.validation.property_validator import PropertyValidator
from src.data_processing.validation.ndvi_validator import NDVIValidator
from src.data_processing.processing.ndvi_extractor import NDVIExtractor

@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    max_workers: int = 4
    chunk_size: int = 100
    scale_factor: float = 0.0001
    required_fields: List[str] = None
    expected_types: Dict[str, str] = None

class BatchProcessor:
    """Handles batch processing of property and NDVI data."""
    
    def __init__(self, config: BatchConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize the batch processor.
        
        Args:
            config: BatchConfig object containing processing parameters
            logger: Optional logger instance. If None, creates a new one.
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.error_handler = ErrorHandler(self.logger)
        
        # Initialize components
        self.property_loader = PropertyLoader(self.logger)
        self.ndvi_loader = NDVILoader(self.logger)
        self.property_validator = PropertyValidator(self.logger)
        self.ndvi_validator = NDVIValidator(self.logger)
        self.ndvi_extractor = NDVIExtractor(self.logger)
        
    def process_batch(self,
                     property_file: Path,
                     ndvi_dir: Path,
                     output_dir: Path,
                     years: List[int]) -> Dict[str, Any]:
        """
        Process a batch of properties and NDVI data.
        
        Args:
            property_file: Path to property data file
            ndvi_dir: Directory containing NDVI files
            output_dir: Directory for output files
            years: List of years to process
            
        Returns:
            Dictionary containing processing results and statistics
            
        Raises:
            ProcessingError: If batch processing fails
        """
        try:
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load and validate property data
            properties = self._load_and_validate_properties(property_file)
            
            # Load and validate NDVI files
            ndvi_files = self._load_and_validate_ndvi(ndvi_dir, years)
            
            # Process in chunks
            results = self._process_chunks(properties, ndvi_files)
            
            # Save results
            self._save_results(results, output_dir)
            
            return self._generate_summary(results)
            
        except Exception as e:
            raise self.error_handler.handle_processing_error(
                error=e,
                context="batch processing"
            )
    
    def _load_and_validate_properties(self, property_file: Path) -> gpd.GeoDataFrame:
        """Load and validate property data."""
        properties = self.property_loader.load_property_data(property_file)
        properties = self.property_loader.ensure_crs(properties)
        properties = self.property_loader.clean_property_data(properties)
        
        if self.config.required_fields:
            self.property_validator.validate_required_fields(
                properties,
                self.config.required_fields
            )
            
        if self.config.expected_types:
            self.property_validator.validate_data_types(
                properties,
                self.config.expected_types
            )
            
        properties = self.property_validator.validate_geometries(properties)
        
        return properties
    
    def _load_and_validate_ndvi(self,
                               ndvi_dir: Path,
                               years: List[int]) -> List[Path]:
        """Load and validate NDVI files."""
        ndvi_files_by_year = self.ndvi_loader.load_ndvi_files(ndvi_dir, years)
        
        # Flatten the list of files
        ndvi_files = []
        for year_files in ndvi_files_by_year.values():
            ndvi_files.extend(year_files)
            
        # Validate files
        metadata_list = []
        valid_files = []
        
        for file_path in ndvi_files:
            try:
                metadata = self.ndvi_validator.validate_ndvi_file(file_path)
                metadata_list.append(metadata)
                valid_files.append(file_path)
            except Exception as e:
                self.logger.warning(f"Skipping invalid NDVI file {file_path}: {str(e)}")
                
        # Validate temporal consistency
        self.ndvi_validator.validate_temporal_consistency(metadata_list)
        
        return valid_files
    
    def _process_chunks(self,
                       properties: gpd.GeoDataFrame,
                       ndvi_files: List[Path]) -> List[pd.DataFrame]:
        """Process properties in chunks using parallel processing."""
        results = []
        
        # Split properties into chunks
        chunks = [properties[i:i + self.config.chunk_size] 
                 for i in range(0, len(properties), self.config.chunk_size)]
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_chunk = {
                executor.submit(
                    self.ndvi_extractor.process_property_batch,
                    chunk,
                    ndvi_files,
                    self.config.scale_factor
                ): i for i, chunk in enumerate(chunks)
            }
            
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    chunk_results = future.result()
                    if not chunk_results.empty:
                        results.append(chunk_results)
                        self.logger.info(f"Completed chunk {chunk_idx}")
                except Exception as e:
                    self.logger.error(f"Error processing chunk {chunk_idx}: {str(e)}")
                    
        return results
    
    def _save_results(self, results: List[pd.DataFrame], output_dir: Path) -> None:
        """Save processing results."""
        if not results:
            self.logger.warning("No results to save")
            return
            
        # Combine all results
        combined_results = pd.concat(results, ignore_index=True)
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"ndvi_results_{timestamp}.csv"
        combined_results.to_csv(output_file, index=False)
        
        self.logger.info(f"Saved results to {output_file}")
    
    def _generate_summary(self, results: List[pd.DataFrame]) -> Dict[str, Any]:
        """Generate processing summary statistics."""
        if not results:
            return {
                "status": "completed",
                "properties_processed": 0,
                "total_observations": 0,
                "success_rate": 0.0
            }
            
        combined_results = pd.concat(results, ignore_index=True)
        
        return {
            "status": "completed",
            "properties_processed": combined_results['property_id'].nunique(),
            "total_observations": len(combined_results),
            "success_rate": len(combined_results) / 
                          (combined_results['property_id'].nunique() * 
                           combined_results['date'].nunique()),
            "date_range": {
                "start": combined_results['date'].min().strftime("%Y-%m-%d"),
                "end": combined_results['date'].max().strftime("%Y-%m-%d")
            },
            "ndvi_stats": {
                "mean": combined_results['mean_ndvi'].mean(),
                "std": combined_results['mean_ndvi'].std(),
                "min": combined_results['mean_ndvi'].min(),
                "max": combined_results['mean_ndvi'].max()
            }
        } 