"""
GridFIA: Spatial Raster Analysis for USDA Forest Service BIGMAP Data

Part of the FIA Python Ecosystem:
- PyFIA: Survey/plot data analysis (https://github.com/mihiarc/pyfia)
- GridFIA: Spatial raster analysis (this package)
- PyFVS: Growth/yield simulation (https://github.com/mihiarc/pyfvs)
- AskFIA: AI conversational interface (https://github.com/mihiarc/askfia)

GridFIA provides efficient access to BIGMAP 2018 forest biomass data at 30m
resolution, with Zarr-based storage and comprehensive diversity metrics.
"""

__version__ = "0.5.0"
__author__ = "Christopher Mihiar"
__email__ = "christopher.mihiar@usda.gov"
__license__ = "MIT"

# Main API - this is the primary interface
from gridfia.api import GridFIA

# Configuration management for advanced users
from gridfia.config import (
    GridFIASettings,
    CloudStorageConfig,
    CloudStorageBackend,
    load_settings,
    save_settings,
)

# Utility classes for advanced users
from gridfia.utils.zarr_utils import ZarrStore

# Domain-specific exceptions
from gridfia.exceptions import (
    GridFIAException,
    InvalidZarrStructure,
    SpeciesNotFound,
    CalculationFailed,
    APIConnectionError,
    InvalidLocationConfig,
    DownloadError,
)

# Backwards compatibility aliases (deprecated, will be removed in v1.0)
BigMapAPI = GridFIA
BigMapSettings = GridFIASettings

# Define what gets imported with "from gridfia import *"
__all__ = [
    # Main API (primary interface)
    "GridFIA",

    # Configuration management
    "GridFIASettings",
    "CloudStorageConfig",
    "CloudStorageBackend",
    "load_settings",
    "save_settings",

    # Utility classes
    "ZarrStore",

    # Exceptions
    "GridFIAException",
    "InvalidZarrStructure",
    "SpeciesNotFound",
    "CalculationFailed",
    "APIConnectionError",
    "InvalidLocationConfig",
    "DownloadError",

    # Backwards compatibility (deprecated)
    "BigMapAPI",
    "BigMapSettings",

    # Package metadata
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]


def get_version() -> str:
    """Return the package version."""
    return __version__


def get_package_info() -> dict:
    """Return package information as a dictionary."""
    return {
        "name": "gridfia",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "license": __license__,
        "description": "Spatial raster analysis for USDA Forest Service BIGMAP data",
        "ecosystem": "FIA Python Ecosystem",
    }
