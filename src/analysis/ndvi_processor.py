#!/usr/bin/env python3

import os
from pathlib import Path
import geopandas as gpd
import pandas as pd
import logging
from typing import Optional

from .mosaic_creator import MosaicCreator
from .ndvi_trend import NDVITrendAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_ndvi(
    ndvi_dir: str = "data/raw/ndvi",
    output_dir: str = "output/ndvi",
    matches_file: str = "output/matches/vance_matches.csv",
    processed_dir: str = "output/processed",
    n_workers: Optional[int] = None,
    save_mosaics: bool = False
) -> None:
    """
    Process NDVI data for matched properties.
    
    Args:
        ndvi_dir: Directory containing NDVI files
        output_dir: Directory to save results
        matches_file: Path to property matches file
        processed_dir: Directory containing processed property data
        n_workers: Number of worker processes
        save_mosaics: Whether to save final mosaic files
    """
    # Create output directories
    mosaic_dir = os.path.join(output_dir, "mosaics")
    trends_dir = os.path.join(output_dir, "trends")
    os.makedirs(mosaic_dir, exist_ok=True)
    os.makedirs(trends_dir, exist_ok=True)
    
    # Load property data
    logger.info("Loading property data...")
    heirs = pd.read_parquet(os.path.join(processed_dir, "heirs_processed.parquet"))
    parcels = pd.read_parquet(os.path.join(processed_dir, "parcels_processed.parquet"))
    logger.info(f"Loaded {len(heirs)} heirs properties and {len(parcels)} parcels")
    
    # Load matches
    logger.info("Loading property matches...")
    matches = pd.read_csv(matches_file)
    logger.info(f"Loaded {len(matches)} property matches")
    
    # Get unique properties to process
    property_ids = pd.concat([
        matches['heirs_property_id'],
        matches['matched_property_id']
    ]).unique()
    logger.info(f"Found {len(property_ids)} unique properties to process")
    
    # Create property GeoDataFrame
    all_properties = pd.concat([heirs, parcels])
    properties_to_process = all_properties[all_properties.index.isin(property_ids)]
    properties_gdf = gpd.GeoDataFrame(properties_to_process)
    logger.info(f"Prepared {len(properties_gdf)} properties for processing")
    
    # Step 1: Create mosaics
    logger.info("Creating NDVI mosaics...")
    with MosaicCreator(ndvi_dir, mosaic_dir) as mosaic_creator:
        mosaic_paths = mosaic_creator.create_all_mosaics(save_final=save_mosaics)
    logger.info(f"Created mosaics for years: {', '.join(sorted(mosaic_paths.keys()))}")
    
    # Step 2: Calculate NDVI trends
    logger.info("Calculating NDVI trends...")
    trend_analyzer = NDVITrendAnalyzer(mosaic_paths, trends_dir, n_workers)
    results_df = trend_analyzer.process_properties(properties_gdf)
    logger.info(f"Completed NDVI trend analysis for {len(results_df)} properties")
    
    # Log summary statistics
    valid_trends = results_df[results_df['slope'].notna()]
    logger.info(f"Generated valid trends for {len(valid_trends)} properties")
    logger.info(f"Average years covered: {valid_trends['years_covered'].mean():.1f}")
    logger.info(f"Average total change: {valid_trends['total_change'].mean():.3f}")
    
if __name__ == "__main__":
    process_ndvi() 