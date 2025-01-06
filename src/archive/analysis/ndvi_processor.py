#!/usr/bin/env python3

import os
from pathlib import Path
import geopandas as gpd
import pandas as pd
import logging
from typing import Optional
import rasterio
from shapely.geometry import box

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
    
    # Get combined NDVI bounds from parts 1 and 2 (using 2018 as reference year)
    logger.info("Getting NDVI coverage bounds from raster parts...")
    ndvi_files = []
    for part in ['p1', 'p2']:
        pattern = f"ndvi_NAIP_Vance_County_2018-{part}.tif"
        matches = list(Path(ndvi_dir).glob(pattern))
        ndvi_files.extend(matches)
    
    if not ndvi_files:
        raise FileNotFoundError(f"No NDVI files found in {ndvi_dir}")
    
    logger.info(f"Found {len(ndvi_files)} NDVI raster parts")
    
    # Get combined bounds from both parts
    bounds_list = []
    ndvi_crs = None
    for ndvi_file in ndvi_files:
        with rasterio.open(ndvi_file) as src:
            bounds_list.append(src.bounds)
            if ndvi_crs is None:
                ndvi_crs = src.crs
            elif ndvi_crs != src.crs:
                raise ValueError(f"Inconsistent CRS found: {ndvi_file} has {src.crs}, expected {ndvi_crs}")
    
    # Calculate combined bounds
    combined_bounds = rasterio.coords.BoundingBox(
        left=min(b.left for b in bounds_list),
        bottom=min(b.bottom for b in bounds_list),
        right=max(b.right for b in bounds_list),
        top=max(b.top for b in bounds_list)
    )
    logger.info(f"Combined NDVI bounds: {combined_bounds}")
    
    # Load only Vance County heirs properties first
    logger.info("Loading Vance County heirs properties...")
    heirs = pd.read_parquet(os.path.join(processed_dir, "heirs_processed.parquet"))
    heirs = heirs[heirs['county_nam'].str.contains('Vance', case=False, na=False)]
    heirs_gdf = gpd.GeoDataFrame(heirs, geometry='geometry', crs="EPSG:2264")
    heirs_gdf = heirs_gdf.to_crs(ndvi_crs)
    
    # Filter heirs properties by combined NDVI bounds
    bounds_poly = box(combined_bounds.left, combined_bounds.bottom, 
                     combined_bounds.right, combined_bounds.top)
    heirs_in_bounds = heirs_gdf[heirs_gdf.intersects(bounds_poly)]
    logger.info(f"Found {len(heirs_in_bounds)} heirs properties within NDVI coverage")
    
    # Load matches only for properties within bounds
    logger.info("Loading property matches...")
    matches = pd.read_csv(matches_file)
    matches = matches[matches['heir_id'].isin(heirs_in_bounds['property_id'])]
    logger.info(f"Found {len(matches)} matches for properties within NDVI coverage")
    
    # Load only the necessary neighbor parcels
    logger.info("Loading matched neighbor parcels...")
    neighbor_ids = matches['neighbor_id'].unique()
    parcels = pd.read_parquet(
        os.path.join(processed_dir, "parcels_processed.parquet"),
        filters=[('property_id', 'in', neighbor_ids.tolist())]
    )
    logger.info(f"Loaded {len(parcels)} neighbor parcels")
    
    # Create property GeoDataFrame for processing
    properties_to_process = pd.concat([heirs_in_bounds, parcels])
    properties_gdf = gpd.GeoDataFrame(properties_to_process, geometry='geometry', crs=ndvi_crs)
    logger.info(f"Prepared {len(properties_gdf)} total properties for processing")
    
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