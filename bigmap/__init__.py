"""
BigMap: North Carolina Forest Biomass and Species Diversity Analysis Tools

A comprehensive Python package for accessing and analyzing forest biomass 
and species diversity data from the BIGMAP 2018 dataset via REST API.
Focus on direct data retrieval and statistical analysis workflows.
"""

__version__ = "0.1.0"
__author__ = "Christopher Mihiar"
__email__ = "christopher.mihiar@usda.gov"
__license__ = "MIT"

# Import REST API functionality
from bigmap.api import BigMapRestClient

# Import configuration and console output
from bigmap.config import BigMapSettings, settings, load_settings, save_settings
from bigmap.console import console, print_success, print_error, print_warning, print_info

# Import calculation framework (for API-based calculations)
from bigmap.core.calculations import registry as calculation_registry

# Define what gets imported with "from bigmap import *"
__all__ = [
    # API access (primary functionality)
    "BigMapRestClient",
    # Calculation framework
    "calculation_registry",
    # Configuration management
    "BigMapSettings",
    "load_settings",
    "save_settings",
    # Console utilities
    "console",
    "print_success",
    "print_error", 
    "print_warning",
    "print_info",
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
        "name": "bigmap",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "license": __license__,
        "description": "North Carolina forest biomass and species diversity analysis tools via REST API",
    }
