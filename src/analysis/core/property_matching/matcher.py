#!/usr/bin/env python3

from dataclasses import dataclass
from pathlib import Path
import logging
from typing import List, Tuple, Optional

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MatchingConfig:
    """Configuration for property matching."""
    min_size_ratio: float = 0.6
    max_size_ratio: float = 1.4
    distance_threshold: float = 2000  # meters
    n_workers: Optional[int] = None

class PropertyMatcher:
    """Matches heirs properties with comparable non-heirs properties."""
    
    def __init__(self, config: MatchingConfig):
        """Initialize the property matcher.
        
        Args:
            config: MatchingConfig object with matching parameters
        """
        self.config = config
        self.heirs_properties = None
        self.all_properties = None
        
    def load_data(self, processed_dir: Path = Path("data/processed")) -> None:
        """Load preprocessed property data.
        
        Args:
            processed_dir: Directory containing processed property data
        """
        heirs_path = processed_dir / "heirs_processed.parquet"
        parcels_path = processed_dir / "parcels_processed.parquet"
        
        logger.info("Loading processed property data...")
        self.heirs_properties = gpd.read_parquet(heirs_path)
        self.all_properties = gpd.read_parquet(parcels_path)
        
        logger.info(f"Loaded {len(self.heirs_properties)} heirs properties and "
                   f"{len(self.all_properties)} total properties")

    def find_matches(self, heir_property: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Find matching properties for a single heirs property.
        
        Args:
            heir_property: Single heirs property as GeoDataFrame row
            
        Returns:
            GeoDataFrame of matching properties
        """
        # Calculate distance to all properties
        heir_centroid = heir_property.geometry.centroid
        distances = self.all_properties.geometry.centroid.distance(heir_centroid)
        
        # Filter by distance threshold
        nearby = self.all_properties[distances <= self.config.distance_threshold].copy()
        
        if len(nearby) == 0:
            return gpd.GeoDataFrame()
            
        # Calculate area ratios
        heir_area = heir_property.geometry.area
        area_ratios = nearby.geometry.area / heir_area
        
        # Filter by size ratio
        size_matches = nearby[
            (area_ratios >= self.config.min_size_ratio) & 
            (area_ratios <= self.config.max_size_ratio)
        ]
        
        return size_matches

    def match_all_properties(self) -> pd.DataFrame:
        """Find matches for all heirs properties.
        
        Returns:
            DataFrame with matched properties
        """
        if self.heirs_properties is None or self.all_properties is None:
            raise ValueError("Must load data before matching properties")
            
        matches = []
        
        for idx, heir in self.heirs_properties.iterrows():
            matching_properties = self.find_matches(heir)
            
            if len(matching_properties) > 0:
                for _, match in matching_properties.iterrows():
                    matches.append({
                        'heir_id': heir.id,
                        'match_id': match.id,
                        'distance': heir.geometry.centroid.distance(match.geometry.centroid),
                        'area_ratio': match.geometry.area / heir.geometry.area
                    })
                    
        return pd.DataFrame(matches)

    def save_matches(self, matches: pd.DataFrame, output_dir: Path = Path("data/processed")) -> None:
        """Save property matches to file.
        
        Args:
            matches: DataFrame of property matches
            output_dir: Directory to save matches
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "property_matches.parquet"
        matches.to_parquet(output_path)
        logger.info(f"Saved {len(matches)} matches to {output_path}") 