"""
Utility functions for GridFIA.

This module contains infrastructure and helper utilities that don't fit
into the core processing, analysis, or ETL categories.
"""

from .parallel_processing import ParallelProcessor
from .zarr_utils import (
    ZarrStore,
    create_zarr_from_geotiffs,
    create_expandable_zarr_from_base_raster,
    append_species_to_zarr,
    batch_append_species_from_dir,
    validate_zarr_store,
)

__all__ = [
    'ParallelProcessor',
    'ZarrStore',
    'create_zarr_from_geotiffs',
    'create_expandable_zarr_from_base_raster',
    'append_species_to_zarr',
    'batch_append_species_from_dir',
    'validate_zarr_store',
]