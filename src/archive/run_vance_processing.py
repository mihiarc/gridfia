"""
Vance County Data Processing Runner

This script orchestrates the processing of Vance County property and NDVI data.
It handles the complete data processing pipeline, including:
1. Property filtering and validation
2. NDVI data extraction and processing
3. Preparation of processed data for subsequent analysis
"""
from pathlib import Path
import logging
import argparse
import sys

from data_processing.config.vance_config import (
    DATA_DIR,
    OUTPUT_DIR,
    STANDARD_CRS,
    LOG_FORMAT,
    LOG_LEVEL,
    LOG_FILE
)
from data_processing.properties.vance_filter import VancePropertyFilter
from data_processing.ndvi.vance_processor import VanceNDVIProcessor

def setup_logging():
    """Configure logging for the data processing pipeline."""
    log_file = OUTPUT_DIR / LOG_FILE
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Create formatters and add it to handlers
    formatter = logging.Formatter(LOG_FORMAT)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Log initial information
    logger = logging.getLogger(__name__)
    logger.info("Starting Vance County Data Processing")
    logger.info(f"Using standard CRS: {STANDARD_CRS}")
    logger.info(f"Log file: {log_file}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Vance County data processing pipeline'
    )
    parser.add_argument(
        '--heirs-file',
        type=Path,
        required=True,
        help='Path to heirs property data'
    )
    parser.add_argument(
        '--parcels-file',
        type=Path,
        required=True,
        help='Path to NC parcels data'
    )
    parser.add_argument(
        '--skip-filtering',
        action='store_true',
        help='Skip property filtering if already done'
    )
    return parser.parse_args()

def main():
    """Run the complete Vance County data processing pipeline."""
    # Setup logging first
    setup_logging()
    logger = logging.getLogger(__name__)
    
    args = parse_args()
    
    try:
        # Step 1: Filter Properties
        filtered_properties_path = DATA_DIR / "vance_properties.parquet"
        if not args.skip_filtering or not filtered_properties_path.exists():
            logger.info("Starting property filtering...")
            property_filter = VancePropertyFilter(
                args.heirs_file,
                args.parcels_file
            )
            filtered_properties_path = property_filter.run_filtering()
        else:
            logger.info("Using existing filtered properties...")
        
        # Step 2: Process NDVI
        logger.info("Starting NDVI data extraction...")
        ndvi_processor = VanceNDVIProcessor(filtered_properties_path)
        ndvi_stats = ndvi_processor.process_all_properties()
        
        # Step 3: Calculate Trends
        logger.info("Calculating initial NDVI trends...")
        results = ndvi_processor.calculate_trends(ndvi_stats)
        
        # Save Results
        output_file = OUTPUT_DIR / "vance_processed_ndvi.parquet"
        logger.info(f"Saving processed data to {output_file}")
        results.to_parquet(output_file)
        
        # Log Processing Summary
        logger.info("\nProcessing Summary:")
        logger.info(f"Total properties processed: {len(results)}")
        logger.info(f"All processing completed in {STANDARD_CRS}")
        
        logger.info("\nInitial NDVI Statistics:")
        for year in [2018, 2020, 2022]:
            mean_ndvi = results[f'ndvi_{year}_mean'].mean()
            std_ndvi = results[f'ndvi_{year}_mean'].std()
            logger.info(f"{year}: Mean NDVI = {mean_ndvi:.3f} (±{std_ndvi:.3f})")
        
        logger.info("\nInitial Trend Statistics:")
        mean_slope = results['ndvi_trend_slope'].mean()
        mean_r2 = results['ndvi_trend_r2'].mean()
        logger.info(f"Mean trend slope: {mean_slope:.5f}")
        logger.info(f"Mean R² value: {mean_r2:.3f}")
        
        logger.info("\nData processing completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Data processing failed: {e}", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main()) 