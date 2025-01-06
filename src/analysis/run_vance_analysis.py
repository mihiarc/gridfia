"""
Vance County Prototype Analysis Runner
"""
from pathlib import Path
import logging
import argparse
import sys

from config.vance_config import DATA_DIR, OUTPUT_DIR
from properties.vance_filter import VancePropertyFilter
from ndvi.vance_processor import VanceNDVIProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / 'vance_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Vance County prototype analysis'
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
    """Run the complete Vance County analysis."""
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
        logger.info("Starting NDVI processing...")
        ndvi_processor = VanceNDVIProcessor(filtered_properties_path)
        ndvi_stats = ndvi_processor.process_all_properties()
        
        # Step 3: Calculate Trends
        logger.info("Calculating NDVI trends...")
        results = ndvi_processor.calculate_trends(ndvi_stats)
        
        # Save Results
        output_file = OUTPUT_DIR / "vance_ndvi_results.parquet"
        logger.info(f"Saving results to {output_file}")
        results.to_parquet(output_file)
        
        # Log Summary Statistics
        logger.info("\nAnalysis Summary:")
        logger.info(f"Total properties processed: {len(results)}")
        logger.info("\nNDVI Statistics:")
        for year in [2018, 2020, 2022]:
            mean_ndvi = results[f'ndvi_{year}_mean'].mean()
            std_ndvi = results[f'ndvi_{year}_mean'].std()
            logger.info(f"{year}: Mean NDVI = {mean_ndvi:.3f} (±{std_ndvi:.3f})")
        
        logger.info("\nTrend Statistics:")
        mean_slope = results['ndvi_trend_slope'].mean()
        mean_r2 = results['ndvi_trend_r2'].mean()
        logger.info(f"Mean trend slope: {mean_slope:.5f}")
        logger.info(f"Mean R² value: {mean_r2:.3f}")
        
        logger.info("\nAnalysis completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main()) 