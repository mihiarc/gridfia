#!/usr/bin/env python3
"""Command-line interface for processing NDVI data for neighbor properties."""

import argparse
import logging
from pathlib import Path
import multiprocessing as mp

from .neighbor_processor import NeighborNDVIProcessor
from src.config.loader import get_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)8s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process NDVI data for neighbor properties"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/paths.yaml"),
        help="Path to configuration file (default: config/paths.yaml)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count - 1)"
    )
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configuration
    config = get_config(args.config)
    
    # Set default number of workers
    if args.workers is None:
        args.workers = max(1, mp.cpu_count() - 1)
    
    try:
        # Initialize and run processor
        processor = NeighborNDVIProcessor(
            ndvi_dir=config.data.input.ndvi.vance_county_dir,
            output_dir=config.data.output.base_dir / "ndvi/neighbors",
            matches_file=config.data.output.base_dir / config.patterns.property_matches,
            processed_dir=config.data.processed.base_dir,
            n_workers=args.workers
        )
        
        result_file = processor.process()
        
        if result_file:
            logger.info(f"Successfully processed neighbor NDVI data")
            logger.info(f"Results saved to: {result_file}")
        else:
            logger.warning("No results generated")
            
    except Exception as e:
        logger.error(f"Error processing neighbor NDVI data: {str(e)}")
        raise

if __name__ == "__main__":
    main() 