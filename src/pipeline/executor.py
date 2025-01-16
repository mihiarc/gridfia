#!/usr/bin/env python3

import argparse
from pathlib import Path
import logging
import yaml
from typing import List, Optional

from .stages import Pipeline, PipelineStage, PipelineConfig
from ..core.property_matching.matcher import MatchingConfig
from ..core.property_matching.validation import ValidationConfig
from ..core.ndvi.processor import NDVIConfig
from ..core.stats.analyzer import StatsConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: Path) -> PipelineConfig:
    """Load pipeline configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        PipelineConfig object
    """
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
        
    return PipelineConfig(
        matching=MatchingConfig(**config_dict.get('matching', {})),
        validation=ValidationConfig(**config_dict.get('validation', {})),
        ndvi=NDVIConfig(**config_dict.get('ndvi', {})),
        stats=StatsConfig(**config_dict.get('stats', {})),
        input_dir=Path(config_dict.get('input_dir', 'data/raw')),
        output_dir=Path(config_dict.get('output_dir', 'data/processed')),
        ndvi_pattern=config_dict.get('ndvi_pattern', 'ndvi_*.tif')
    )

def parse_stages(stages_str: Optional[str]) -> Optional[List[PipelineStage]]:
    """Parse pipeline stages from comma-separated string.
    
    Args:
        stages_str: Comma-separated string of stage names
        
    Returns:
        List of PipelineStage enums
    """
    if not stages_str:
        return None
        
    try:
        return [PipelineStage[s.strip().upper()] for s in stages_str.split(',')]
    except KeyError as e:
        raise ValueError(f"Invalid stage name: {e}")

def main():
    """Main entry point for pipeline execution."""
    parser = argparse.ArgumentParser(
        description='Execute heirs property analysis pipeline'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('config/analysis.yaml'),
        help='Path to pipeline configuration file'
    )
    parser.add_argument(
        '--stages',
        type=str,
        help='Comma-separated list of stages to run (runs all if not specified)'
    )
    parser.add_argument(
        '--log-file',
        type=Path,
        help='Path to log file (logs to console if not specified)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logging.getLogger().addHandler(file_handler)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        
        # Parse stages
        stages = parse_stages(args.stages)
        if stages:
            logger.info(f"Running stages: {', '.join(s.value for s in stages)}")
        else:
            logger.info("Running all pipeline stages")
        
        # Initialize and run pipeline
        pipeline = Pipeline(config)
        results = pipeline.run(stages)
        
        logger.info("Pipeline completed successfully")
        
        # Log summary of results
        for stage, result in results.items():
            if isinstance(result, dict):
                logger.info(f"{stage} results summary:")
                for key, value in result.items():
                    if isinstance(value, (int, float, str)):
                        logger.info(f"  {key}: {value}")
            else:
                logger.info(f"{stage} completed with {type(result).__name__} output")
                
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main() 