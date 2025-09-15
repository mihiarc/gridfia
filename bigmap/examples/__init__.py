"""
BigMap examples utilities.

This subpackage contains utilities specifically for running the BigMap examples.
These are separated from the main API to maintain clean separation of concerns.
"""

from bigmap.examples.utils import (
    AnalysisConfig,
    cleanup_example_outputs,
    safe_download_species,
    safe_load_zarr_with_memory_check,
    create_zarr_from_rasters,
    create_sample_zarr,
    print_zarr_info,
    calculate_basic_stats,
    validate_species_codes,
    add_zarr_metadata
)

__all__ = [
    "AnalysisConfig",
    "cleanup_example_outputs",
    "safe_download_species",
    "safe_load_zarr_with_memory_check",
    "create_zarr_from_rasters",
    "create_sample_zarr",
    "print_zarr_info",
    "calculate_basic_stats",
    "validate_species_codes",
    "add_zarr_metadata",
]