#!/usr/bin/env python3

import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import sys
from stats_analyzer import StatsAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def extract_ndvi_values(args):
    """Extract NDVI values for a geometry from raster files."""
    geometry, year, ndvi_dir = args
    try:
        values = []
        for part in [1, 2]:
            raster_path = ndvi_dir / f"ndvi_NAIP_Vance_County_{year}-p{part}.tif"
            logger.debug(f"Processing raster: {raster_path}")
            with rasterio.open(raster_path) as src:
                # Mask the raster with the geometry
                out_image, _ = mask(src, [geometry], crop=True, all_touched=True)
                
                # Get valid values (exclude nodata)
                valid_values = out_image[0][out_image[0] != src.nodata]
                if len(valid_values) > 0:
                    values.append(np.nanmean(valid_values))
                    logger.debug(f"Extracted mean NDVI value: {values[-1]}")
        
        result = np.mean(values) if values else np.nan
        logger.debug(f"Final NDVI value for year {year}: {result}")
        return result
            
    except Exception as e:
        logger.warning(f"Error extracting NDVI values for year {year}: {e}")
        return np.nan

def process_property(args):
    """Process NDVI values for a single property."""
    idx, geometry, years, ndvi_dir = args
    try:
        logger.debug(f"Processing property {idx}")
        values = {}
        for year in years:
            value = extract_ndvi_values((geometry, year, ndvi_dir))
            values[f'ndvi_{year}'] = value
            
        # Calculate trend slope
        ndvi_values = [values[f'ndvi_{year}'] for year in years]
        if all(pd.notna(v) for v in ndvi_values):
            values['ndvi_trend_slope'] = (ndvi_values[-1] - ndvi_values[0]) / (years[-1] - years[0])
            logger.debug(f"Calculated trend slope for property {idx}: {values['ndvi_trend_slope']}")
        else:
            values['ndvi_trend_slope'] = np.nan
            logger.debug(f"Could not calculate trend slope for property {idx} due to missing values")
            
        return idx, values
        
    except Exception as e:
        logger.warning(f"Error processing property {idx}: {e}")
        return idx, {f'ndvi_{year}': np.nan for year in years} | {'ndvi_trend_slope': np.nan}

def load_and_prepare_data():
    """Load and prepare NDVI data for analysis."""
    try:
        logger.info("Starting data preparation...")
        
        # Load processed data
        logger.debug("Loading property data...")
        heirs_data = gpd.read_parquet("output/processed/heirs_processed.parquet")
        parcels_data = gpd.read_parquet("output/processed/parcels_processed.parquet")
        
        logger.info(f"Loaded {len(heirs_data)} heirs properties and {len(parcels_data)} parcels")
        
        # Filter to Vance County
        logger.debug("Filtering to Vance County...")
        vance_heirs = heirs_data[heirs_data['county_nam'] == 'Vance']
        vance_parcels = parcels_data[parcels_data['FIPS'] == '37181']  # Vance County FIPS
        
        logger.info(f"Filtered to {len(vance_heirs)} Vance County heirs properties and {len(vance_parcels)} parcels")
        
        # Prepare combined dataset
        logger.debug("Preparing combined dataset...")
        vance_heirs['is_heir'] = True
        vance_parcels['is_heir'] = False
        
        # Combine datasets
        combined_data = pd.concat([vance_heirs, vance_parcels], ignore_index=True)
        logger.debug(f"Combined dataset has {len(combined_data)} properties")
        
        # Process NDVI values in parallel
        years = [2018, 2020, 2022]
        ndvi_dir = Path("data/raw/ndvi")
        
        # Prepare arguments for parallel processing
        process_args = [
            (idx, row.geometry, years, ndvi_dir)
            for idx, row in combined_data.iterrows()
        ]
        
        # Process properties in parallel with progress bar
        logger.info("Processing NDVI data in parallel...")
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_property, args) for args in process_args]
            
            results = {}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing properties"):
                idx, values = future.result()
                results[idx] = values
                logger.debug(f"Completed processing property {idx}")
        
        # Update combined data with results
        logger.debug("Updating dataset with NDVI results...")
        for idx, values in results.items():
            for key, value in values.items():
                combined_data.loc[idx, key] = value
        
        # Save prepared data
        output_file = "output/analysis/ndvi_analysis_data.parquet"
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        combined_data.to_parquet(output_file)
        logger.info(f"Saved prepared data to {output_file}")
        
        return output_file
        
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        raise

def main():
    """Run NDVI trend analysis."""
    try:
        logger.info("Starting NDVI trend analysis...")
        
        # Prepare data
        data_file = load_and_prepare_data()
        
        # Initialize analyzer
        logger.debug("Initializing StatsAnalyzer...")
        analyzer = StatsAnalyzer(output_dir="output/analysis")
        
        # Run analysis
        logger.info("Running statistical analysis...")
        stats_df = analyzer.run_analysis(data_file)
        
        # Print summary
        print("\nNDVI Trend Analysis Results")
        print("==========================")
        print("\nTrend Statistics:")
        print(stats_df)
        
        print("\nAnalysis outputs saved to output/analysis/")
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        raise

if __name__ == "__main__":
    main() 