"""
Utility functions for BigMap.

This module contains infrastructure and helper utilities that don't fit
into the core processing, analysis, or ETL categories.
"""

from .parallel_processing import ParallelProcessor

__all__ = [
    'ParallelProcessor',
]