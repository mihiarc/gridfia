"""
Data processors and pipelines for forest analysis.

This module contains high-level processors that orchestrate
complex analysis workflows using REST API data.
"""

from .forest_metrics import ForestMetricsProcessor, run_forest_analysis

__all__ = [
    'ForestMetricsProcessor',
    'run_forest_analysis',
]