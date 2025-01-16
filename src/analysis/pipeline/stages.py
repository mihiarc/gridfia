#!/usr/bin/env python3

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import logging
from typing import Dict, List, Optional

import pandas as pd
import geopandas as gpd

from ..core.property_matching.matcher import PropertyMatcher, MatchingConfig
from ..core.property_matching.validation import MatchValidator, ValidationConfig
from ..core.ndvi.processor import NDVIProcessor, NDVIConfig
from ..core.stats.analyzer import StatsAnalyzer, StatsConfig

logger = logging.getLogger(__name__)

class PipelineStage(Enum):
    """Pipeline stages for heirs property analysis."""
    PROPERTY_MATCHING = "property_matching"
    MATCH_VALIDATION = "match_validation"
    NDVI_PROCESSING = "ndvi_processing"
    STATISTICAL_ANALYSIS = "statistical_analysis"

@dataclass
class PipelineConfig:
    """Configuration for analysis pipeline."""
    matching: MatchingConfig
    validation: ValidationConfig
    ndvi: NDVIConfig
    stats: StatsConfig
    input_dir: Path = Path("data/raw")
    output_dir: Path = Path("data/processed")
    ndvi_pattern: str = "ndvi_*.tif"

class Pipeline:
    """Orchestrates the heirs property analysis pipeline."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize the pipeline.
        
        Args:
            config: PipelineConfig object with pipeline parameters
        """
        self.config = config
        self.matcher = PropertyMatcher(config.matching)
        self.validator = MatchValidator(config.validation)
        self.ndvi_processor = NDVIProcessor(config.ndvi)
        self.stats_analyzer = StatsAnalyzer(config.stats)
        
    def run(self, stages: Optional[List[PipelineStage]] = None) -> Dict:
        """Run the analysis pipeline.
        
        Args:
            stages: List of stages to run (runs all if None)
            
        Returns:
            Dictionary with results from each stage
        """
        if stages is None:
            stages = list(PipelineStage)
            
        results = {}
        
        try:
            # Property matching
            if PipelineStage.PROPERTY_MATCHING in stages:
                logger.info("Starting property matching stage...")
                matches = self._run_matching_stage()
                results['property_matching'] = matches
                
            # Match validation
            if PipelineStage.MATCH_VALIDATION in stages:
                logger.info("Starting match validation stage...")
                if 'property_matching' not in results:
                    raise ValueError("Property matching results required for validation")
                valid_matches = self._run_validation_stage(results['property_matching'])
                results['match_validation'] = valid_matches
                
            # NDVI processing
            if PipelineStage.NDVI_PROCESSING in stages:
                logger.info("Starting NDVI processing stage...")
                if 'match_validation' not in results:
                    raise ValueError("Validated matches required for NDVI processing")
                ndvi_results = self._run_ndvi_stage(results['match_validation'])
                results['ndvi_processing'] = ndvi_results
                
            # Statistical analysis
            if PipelineStage.STATISTICAL_ANALYSIS in stages:
                logger.info("Starting statistical analysis stage...")
                if 'ndvi_processing' not in results:
                    raise ValueError("NDVI results required for statistical analysis")
                stats_results = self._run_stats_stage(results['ndvi_processing'])
                results['statistical_analysis'] = stats_results
                
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
            
        return results
    
    def _run_matching_stage(self) -> pd.DataFrame:
        """Run property matching stage.
        
        Returns:
            DataFrame with property matches
        """
        # Load property data
        self.matcher.load_data(self.config.input_dir)
        
        # Find matches
        matches = self.matcher.match_all_properties()
        
        # Save results
        self.matcher.save_matches(matches, self.config.output_dir)
        
        return matches
    
    def _run_validation_stage(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Run match validation stage.
        
        Args:
            matches: DataFrame with property matches
            
        Returns:
            DataFrame with validated matches
        """
        # Validate matches
        valid_matches = self.validator.validate_matches(matches)
        
        # Generate validation report
        self.validator.generate_validation_report(
            valid_matches, self.config.output_dir
        )
        
        return valid_matches
    
    def _run_ndvi_stage(self, matches: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Run NDVI processing stage.
        
        Args:
            matches: DataFrame with validated matches
            
        Returns:
            Dictionary with NDVI results for heirs and matches
        """
        # Load property data
        heirs_properties = gpd.read_parquet(
            self.config.input_dir / "heirs_processed.parquet"
        )
        all_properties = gpd.read_parquet(
            self.config.input_dir / "parcels_processed.parquet"
        )
        
        # Get matched properties
        matched_heirs = heirs_properties[
            heirs_properties['id'].isin(matches['heir_id'])
        ]
        matched_others = all_properties[
            all_properties['id'].isin(matches['match_id'])
        ]
        
        # Process NDVI for each group
        ndvi_files = list(self.config.input_dir.glob(self.config.ndvi_pattern))
        
        heirs_ndvi = []
        matches_ndvi = []
        
        for ndvi_file in ndvi_files:
            heirs_result = self.ndvi_processor.process_property_ndvi(
                matched_heirs, ndvi_file, self.config.output_dir / "ndvi/heirs"
            )
            matches_result = self.ndvi_processor.process_property_ndvi(
                matched_others, ndvi_file, self.config.output_dir / "ndvi/matches"
            )
            
            heirs_ndvi.append(heirs_result)
            matches_ndvi.append(matches_result)
            
        return {
            'heirs': heirs_ndvi,
            'matches': matches_ndvi
        }
    
    def _run_stats_stage(self, ndvi_results: Dict[str, List[pd.DataFrame]]) -> Dict:
        """Run statistical analysis stage.
        
        Args:
            ndvi_results: Dictionary with NDVI results
            
        Returns:
            Dictionary with statistical analysis results
        """
        # Combine NDVI results across dates
        heirs_combined = pd.concat(ndvi_results['heirs'])
        matches_combined = pd.concat(ndvi_results['matches'])
        
        # Run statistical analysis
        stats_results = self.stats_analyzer.analyze_ndvi_differences(
            heirs_combined, matches_combined
        )
        
        # Generate plots
        self.stats_analyzer.generate_plots(
            heirs_combined, matches_combined,
            self.config.output_dir / "plots"
        )
        
        return stats_results 