#!/usr/bin/env python3
"""
Download Lane County forest species data and build Zarr store.

This script:
1. Downloads species raster data from FIA BIGMAP REST API for Lane County, Oregon
2. Converts the downloaded GeoTIFF files into an efficient Zarr store
"""

from pathlib import Path
import logging
from bigmap import BigMapAPI

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    # Configuration
    CONFIG_FILE = Path("configs/counties/lane_oregon.yaml")
    DOWNLOAD_DIR = Path("data/lane_county/downloads")
    ZARR_PATH = Path("data/lane_county/lane_county.zarr")

    # Species codes to download (None = all available species)
    # Common Oregon species you might want to focus on:
    # 0202 - Douglas-fir
    # 0122 - Ponderosa pine
    # 0263 - Western hemlock
    # 0242 - Western redcedar
    # 0081 - Noble fir
    # 0015 - White fir
    # 0119 - Sugar pine
    # 0117 - Lodgepole pine
    SPECIES_CODES = None  # Set to None to download all species
    # SPECIES_CODES = ["0202", "0122", "0263", "0242"]  # Or specify specific species

    # Ensure directories exist
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ZARR_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Initialize API
    api = BigMapAPI()

    # Step 1: Download species data with resume capability
    logger.info("="*60)
    logger.info("Starting Lane County data download...")
    logger.info(f"Config file: {CONFIG_FILE}")
    logger.info(f"Download directory: {DOWNLOAD_DIR}")

    # Check for existing downloads
    existing_files = list(DOWNLOAD_DIR.glob("*.tif")) + list(DOWNLOAD_DIR.glob("*.tiff"))
    if existing_files:
        logger.info(f"Found {len(existing_files)} existing files in download directory")
        logger.info("These will be skipped if already complete")

    # Get list of species to download
    if SPECIES_CODES:
        species_to_download = SPECIES_CODES
        logger.info(f"Downloading specific species: {SPECIES_CODES}")
    else:
        # Get all available species
        all_species = api.list_species()
        species_to_download = [s.species_code for s in all_species]
        logger.info(f"Downloading ALL {len(species_to_download)} available species (this may take a while)")

    # Filter out already downloaded species
    downloaded_files = []
    skipped_count = 0
    species_to_skip = []

    for species_code in species_to_download:
        expected_file = DOWNLOAD_DIR / f"lane_species_{species_code}.tif"
        if expected_file.exists() and expected_file.stat().st_size > 0:
            logger.info(f"  Skipping {species_code} - already downloaded")
            downloaded_files.append(expected_file)
            species_to_skip.append(species_code)
            skipped_count += 1

    if skipped_count > 0:
        logger.info(f"Skipping {skipped_count} already downloaded species")
        remaining = len(species_to_download) - skipped_count
        if remaining > 0:
            logger.info(f"Downloading {remaining} remaining species...")
            # Filter out already downloaded species
            species_to_download = [s for s in species_to_download if s not in species_to_skip]

    try:
        # Only download if there are species to download
        if len(species_to_download) > 0:
            # Pass the filtered list of remaining species to download
            new_downloads = api.download_species(
                location_config=CONFIG_FILE,
                output_dir=DOWNLOAD_DIR,
                species_codes=species_to_download  # Always pass the filtered list
            )
            downloaded_files.extend(new_downloads)

        logger.info(f"Total files available: {len(downloaded_files)} species rasters")
        for f in downloaded_files[:5]:  # Show first 5
            logger.info(f"  - {f.name}")
        if len(downloaded_files) > 5:
            logger.info(f"  ... and {len(downloaded_files) - 5} more files")

    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise

    # Step 2: Build Zarr store (skip if already exists)
    logger.info("="*60)

    # Check if Zarr store already exists
    if ZARR_PATH.exists():
        logger.info(f"Zarr store already exists at: {ZARR_PATH}")
        try:
            # Validate existing store
            info = api.validate_zarr(ZARR_PATH)
            logger.info(f"Existing Zarr store info:")
            logger.info(f"  - Shape: {info['shape']}")
            logger.info(f"  - Number of species: {info['num_species']}")
            logger.info(f"  - Chunk size: {info.get('chunks', 'N/A')}")
            logger.info(f"  - Compression: {info.get('compression', 'N/A')}")

            # Ask user if they want to rebuild
            logger.info("Zarr store already exists and is valid.")
            logger.info("To rebuild, delete the existing store and run again.")
            zarr_path = ZARR_PATH
        except Exception as e:
            logger.warning(f"Existing Zarr store validation failed: {e}")
            logger.info("Rebuilding Zarr store...")

            # Remove invalid store and rebuild
            import shutil
            shutil.rmtree(ZARR_PATH)

            zarr_path = api.create_zarr(
                input_dir=DOWNLOAD_DIR,
                output_path=ZARR_PATH,
                chunk_size=(1, 2000, 2000),  # Lane County is large, use bigger chunks
                compression="lz4",
                compression_level=5,
                include_total=True  # Include total biomass calculation
            )
            logger.info(f"Successfully rebuilt Zarr store at: {zarr_path}")
    else:
        logger.info("Building Zarr store from downloaded GeoTIFFs...")
        logger.info(f"Output path: {ZARR_PATH}")

        try:
            zarr_path = api.create_zarr(
                input_dir=DOWNLOAD_DIR,
                output_path=ZARR_PATH,
                chunk_size=(1, 2000, 2000),  # Lane County is large, use bigger chunks
                compression="lz4",
                compression_level=5,
                include_total=True  # Include total biomass calculation
            )

            logger.info(f"Successfully created Zarr store at: {zarr_path}")

            # Validate the store
            info = api.validate_zarr(zarr_path)
            logger.info(f"Zarr store info:")
            logger.info(f"  - Shape: {info['shape']}")
            logger.info(f"  - Number of species: {info['num_species']}")
            logger.info(f"  - Chunk size: {info.get('chunks', 'N/A')}")
            logger.info(f"  - Compression: {info.get('compression', 'N/A')}")

        except Exception as e:
            logger.error(f"Zarr creation failed: {e}")
            raise

    logger.info("="*60)
    logger.info("Lane County data download and Zarr creation complete!")
    logger.info(f"Zarr store ready for analysis at: {ZARR_PATH}")

    return zarr_path


if __name__ == "__main__":
    main()