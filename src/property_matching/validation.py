#!/usr/bin/env python3

from dataclasses import dataclass
from pathlib import Path
import logging
from typing import Dict, List, Optional

import pandas as pd
import geopandas as gpd
import numpy as np
import json

logger = logging.getLogger(__name__)

@dataclass
class ValidationConfig:
    """Configuration for match validation."""
    min_matches_per_heir: int = 1
    max_matches_per_heir: int = 5
    max_distance_outlier_std: float = 3.0
    max_area_ratio_outlier_std: float = 3.0

class MatchValidator:
    """Validates and filters property matches."""
    
    def __init__(self, config: ValidationConfig):
        """Initialize the match validator.
        
        Args:
            config: ValidationConfig object with validation parameters
        """
        self.config = config
        
    def validate_matches(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Validate and filter property matches.
        
        Args:
            matches: DataFrame of property matches
            
        Returns:
            Filtered DataFrame of valid matches
        """
        if len(matches) == 0:
            logger.warning("No matches to validate")
            return matches
            
        # Remove statistical outliers
        valid_matches = self._remove_outliers(matches)
        
        # Filter by matches per heir
        valid_matches = self._filter_matches_per_heir(valid_matches)
        
        logger.info(f"Validated matches: {len(valid_matches)} of {len(matches)} original matches")
        return valid_matches
        
    def _remove_outliers(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Remove statistical outliers from matches.
        
        Args:
            matches: DataFrame of property matches
            
        Returns:
            DataFrame with outliers removed
        """
        # Calculate z-scores for distance and area ratio
        distance_zscore = np.abs(stats.zscore(matches['distance']))
        area_ratio_zscore = np.abs(stats.zscore(matches['area_ratio']))
        
        # Filter outliers
        valid_matches = matches[
            (distance_zscore <= self.config.max_distance_outlier_std) &
            (area_ratio_zscore <= self.config.max_area_ratio_outlier_std)
        ]
        
        return valid_matches
        
    def _filter_matches_per_heir(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Filter matches to ensure each heir has appropriate number of matches.
        
        Args:
            matches: DataFrame of property matches
            
        Returns:
            Filtered DataFrame
        """
        # Count matches per heir
        matches_per_heir = matches.groupby('heir_id').size()
        
        # Get heirs with valid number of matches
        valid_heirs = matches_per_heir[
            (matches_per_heir >= self.config.min_matches_per_heir) &
            (matches_per_heir <= self.config.max_matches_per_heir)
        ].index
        
        # Filter matches
        valid_matches = matches[matches['heir_id'].isin(valid_heirs)]
        
        return valid_matches
        
    def generate_validation_report(self, matches: pd.DataFrame, 
                                 output_dir: Path = Path("data/processed")) -> Dict:
        """Generate validation report for matches.
        
        Args:
            matches: DataFrame of property matches
            output_dir: Directory to save report
            
        Returns:
            Dictionary with validation statistics
        """
        report = {
            'total_matches': len(matches),
            'unique_heirs': matches['heir_id'].nunique(),
            'unique_matches': matches['match_id'].nunique(),
            'avg_matches_per_heir': matches.groupby('heir_id').size().mean(),
            'avg_distance': matches['distance'].mean(),
            'avg_area_ratio': matches['area_ratio'].mean()
        }
        
        # Save report
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "match_validation_report.json"
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Saved validation report to {output_path}")
        return report 