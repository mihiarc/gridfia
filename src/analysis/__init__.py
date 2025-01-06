"""
Vance County NDVI Analysis Package

This package provides tools for analyzing NDVI data from Vance County properties,
comparing heirs and non-heirs properties. It includes modules for:
- Statistical analysis
- Visualization
- Configuration management
"""

__version__ = '0.1.0'

from .stats.stats_analyzer import NDVIStatsAnalyzer
from .visualization.ndvi_plotter import NDVIPlotter

__all__ = ['NDVIStatsAnalyzer', 'NDVIPlotter'] 