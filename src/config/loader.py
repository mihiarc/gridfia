#!/usr/bin/env python3

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import logging

logger = logging.getLogger(__name__)

@dataclass
class NDVIDataPaths:
    """NDVI data paths configuration."""
    sentinel2_dir: Path
    vance_county_dir: Path

@dataclass
class InputPaths:
    """Input data paths configuration."""
    base_dir: Path
    heirs_properties: Path
    parcels: Path
    ndvi: NDVIDataPaths

@dataclass
class ProcessedPaths:
    """Processed data paths configuration."""
    base_dir: Path
    properties_dir: Path
    ndvi_dir: Path
    prototype_dir: Path

@dataclass
class OutputPaths:
    """Output data paths configuration."""
    base_dir: Path
    plots_dir: Path
    reports_dir: Path

@dataclass
class DataPaths:
    """All data paths configuration."""
    input: InputPaths
    processed: ProcessedPaths
    output: OutputPaths
    temp_dir: Path

@dataclass
class FilePatterns:
    """File pattern configuration."""
    ndvi_files: str
    property_matches: str
    validation_report: str
    ndvi_stats: str

@dataclass
class RequiredFields:
    """Required fields configuration."""
    properties: List[str]
    ndvi: List[str]

@dataclass
class Config:
    """Complete configuration."""
    data: DataPaths
    patterns: FilePatterns
    required_fields: RequiredFields

def load_config(config_path: Path = Path("config/paths.yaml")) -> Config:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config object
    """
    # Load YAML file
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
        
    # Convert paths in data section
    data_dict = config_dict['data']
    
    # Input paths
    input_dict = data_dict['input']
    ndvi_paths = NDVIDataPaths(
        sentinel2_dir=Path(input_dict['ndvi']['sentinel2_dir']),
        vance_county_dir=Path(input_dict['ndvi']['vance_county_dir'])
    )
    input_paths = InputPaths(
        base_dir=Path(input_dict['base_dir']),
        heirs_properties=Path(input_dict['heirs_properties']),
        parcels=Path(input_dict['parcels']),
        ndvi=ndvi_paths
    )
    
    # Processed paths
    processed_dict = data_dict['processed']
    processed_paths = ProcessedPaths(
        base_dir=Path(processed_dict['base_dir']),
        properties_dir=Path(processed_dict['properties_dir']),
        ndvi_dir=Path(processed_dict['ndvi_dir']),
        prototype_dir=Path(processed_dict['prototype_dir'])
    )
    
    # Output paths
    output_dict = data_dict['output']
    output_paths = OutputPaths(
        base_dir=Path(output_dict['base_dir']),
        plots_dir=Path(output_dict['plots_dir']),
        reports_dir=Path(output_dict['reports_dir'])
    )
    
    # Complete data paths
    data_paths = DataPaths(
        input=input_paths,
        processed=processed_paths,
        output=output_paths,
        temp_dir=Path(data_dict['temp_dir'])
    )
    
    # File patterns
    patterns = FilePatterns(**config_dict['patterns'])
    
    # Required fields
    required_fields = RequiredFields(**config_dict['required_fields'])
    
    return Config(
        data=data_paths,
        patterns=patterns,
        required_fields=required_fields
    )

def validate_paths(config: Config) -> None:
    """Validate that required paths exist and create directories if needed.
    
    Args:
        config: Config object to validate
    """
    # Check input files exist
    if not config.data.input.heirs_properties.exists():
        raise FileNotFoundError(f"Heirs properties file not found: {config.data.input.heirs_properties}")
    if not config.data.input.parcels.exists():
        raise FileNotFoundError(f"Parcels file not found: {config.data.input.parcels}")
        
    # Create output directories if they don't exist
    config.data.processed.base_dir.mkdir(parents=True, exist_ok=True)
    config.data.processed.properties_dir.mkdir(parents=True, exist_ok=True)
    config.data.processed.ndvi_dir.mkdir(parents=True, exist_ok=True)
    
    config.data.output.base_dir.mkdir(parents=True, exist_ok=True)
    config.data.output.plots_dir.mkdir(parents=True, exist_ok=True)
    config.data.output.reports_dir.mkdir(parents=True, exist_ok=True)
    
    config.data.temp_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("All required directories created")

def get_config(config_path: Optional[Path] = None) -> Config:
    """Load and validate configuration.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Validated Config object
    """
    # Load configuration
    config = load_config(config_path) if config_path else load_config()
    
    # Validate paths
    validate_paths(config)
    
    return config 