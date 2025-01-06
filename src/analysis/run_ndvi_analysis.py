#!/usr/bin/env python3

import os
import argparse
import logging
from pathlib import Path
import geopandas as gpd
import pandas as pd
from typing import Dict, Optional

from mosaic_creator import MosaicCreator
from ndvi_trend import NDVITrendAnalyzer
from data_integrator import DataIntegrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_mosaics(
    ndvi_dir: str = "data/raw/ndvi",
    output_dir: str = "output/ndvi/mosaics",
    save_final: bool = True
) -> dict:
    """
    Create NDVI mosaics for all years.
    
    Args:
        ndvi_dir: Directory containing NDVI files
        output_dir: Directory to save mosaics
        save_final: Whether to save final GeoTIFF mosaics
        
    Returns:
        Dictionary mapping years to mosaic paths
    """
    logger.info("Starting NDVI mosaic creation...")
    
    # Create mosaic directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create mosaics
    with MosaicCreator(ndvi_dir, output_dir) as mosaic_creator:
        mosaic_paths = mosaic_creator.create_all_mosaics(save_final=save_final)
        
    logger.info(f"Created mosaics for years: {', '.join(sorted(mosaic_paths.keys()))}")
    return mosaic_paths

def calculate_trends(
    mosaic_paths: dict,
    output_dir: str = "output/ndvi/trends",
    matches_file: str = "output/matches/vance_matches.csv",
    processed_dir: str = "output/processed",
    n_workers: int = None
) -> None:
    """
    Calculate NDVI trends for matched properties.
    
    Args:
        mosaic_paths: Dictionary mapping years to mosaic paths
        output_dir: Directory to save trend results
        matches_file: Path to property matches file
        processed_dir: Directory containing processed property data
        n_workers: Number of worker processes
    """
    logger.info("Starting NDVI trend calculation...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize data integrator for validation
    integrator = DataIntegrator()
    
    # Load property data
    logger.info("Loading property data...")
    heirs = pd.read_parquet(os.path.join(processed_dir, "heirs_processed.parquet"))
    parcels = pd.read_parquet(os.path.join(processed_dir, "parcels_processed.parquet"))
    logger.info(f"Loaded {len(heirs)} heirs properties and {len(parcels)} parcels")
    
    # Load matches
    logger.info("Loading property matches...")
    matches = pd.read_csv(matches_file)
    logger.info(f"Loaded {len(matches)} property matches")
    
    # Validate data schemas
    logger.info("Validating data schemas...")
    all_properties = pd.concat([heirs, parcels])
    is_valid, results = integrator.validate_schema(all_properties, 'property')
    if not is_valid:
        logger.warning("Property data validation issues:")
        for k, v in results.items():
            if v:
                logger.warning(f"- {k}: {v}")
    
    is_valid, results = integrator.validate_schema(matches, 'match')
    if not is_valid:
        logger.warning("Match data validation issues:")
        for k, v in results.items():
            if v:
                logger.warning(f"- {k}: {v}")
    
    # Validate relationships
    logger.info("Validating data relationships...")
    rel_results = integrator.validate_relationships(all_properties, matches)
    if rel_results['invalid_references'] or rel_results['duplicate_keys']:
        logger.warning("Relationship validation issues:")
        for k, v in rel_results.items():
            if v:
                logger.warning(f"- {k}: {v}")
    
    # Get unique properties to process
    property_ids = pd.concat([
        matches['heir_id'],
        matches['neighbor_id']
    ]).unique()
    logger.info(f"Found {len(property_ids)} unique properties to process")
    
    # Create property GeoDataFrame
    properties_to_process = all_properties[all_properties['property_id'].isin(property_ids)]
    properties_gdf = gpd.GeoDataFrame(
        properties_to_process,
        geometry=gpd.GeoSeries.from_wkt(properties_to_process['geometry_wkt']),
        crs="EPSG:2264"  # NC State Plane
    )
    logger.info(f"Prepared {len(properties_gdf)} properties for processing")
    
    # Calculate trends
    trend_analyzer = NDVITrendAnalyzer(mosaic_paths, output_dir, n_workers)
    results_df = trend_analyzer.process_properties(properties_gdf)
    
    # Log summary statistics
    valid_trends = results_df[results_df['slope'].notna()]
    logger.info(f"Generated valid trends for {len(valid_trends)} properties")
    logger.info(f"Average years covered: {valid_trends['years_covered'].mean():.1f}")
    logger.info(f"Average total change: {valid_trends['total_change'].mean():.3f}")

def main():
    """Run NDVI analysis steps."""
    parser = argparse.ArgumentParser(description="Run NDVI analysis steps")
    parser.add_argument("--step", choices=["mosaic", "trends", "all"],
                      default="all", help="Analysis step to run")
    parser.add_argument("--ndvi-dir", default="data/raw/ndvi",
                      help="Directory containing NDVI files")
    parser.add_argument("--output-dir", default="output/ndvi",
                      help="Base output directory")
    parser.add_argument("--save-mosaics", action="store_true",
                      help="Save final GeoTIFF mosaics")
    parser.add_argument("--workers", type=int, default=None,
                      help="Number of worker processes")
    
    args = parser.parse_args()
    
    # Create output directories
    mosaic_dir = os.path.join(args.output_dir, "mosaics")
    trends_dir = os.path.join(args.output_dir, "trends")
    
    try:
        if args.step in ["mosaic", "all"]:
            mosaic_paths = create_mosaics(
                args.ndvi_dir,
                mosaic_dir,
                args.save_mosaics
            )
        else:
            # If only running trends, load existing mosaic paths
            mosaic_paths = {}
            for file in os.listdir(mosaic_dir):
                if file.endswith((".vrt", ".tif")):
                    year = file.split("_")[1]
                    mosaic_paths[year] = os.path.join(mosaic_dir, file)
        
        if args.step in ["trends", "all"]:
            calculate_trends(
                mosaic_paths,
                trends_dir,
                n_workers=args.workers
            )
            
    except Exception as e:
        logger.error(f"Error during NDVI analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 