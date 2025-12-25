"""
Configuration management for GridFIA using Pydantic.

This module defines configuration schemas and settings management
for the GridFIA package, part of the FIA Python Ecosystem.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings


# Removed RasterConfig - not needed for REST API approach


class VisualizationConfig(BaseModel):
    """Configuration for visualization parameters."""
    
    default_dpi: int = Field(
        default=300,
        ge=72,
        le=600,
        description="Default DPI for output images"
    )
    default_figure_size: Tuple[float, float] = Field(
        default=(16, 12),
        description="Default figure size in inches (width, height)"
    )
    color_maps: dict = Field(
        default={
            "biomass": "viridis",
            "diversity": "plasma",
            "richness": "Spectral_r"
        },
        description="Default color maps for different data types"
    )
    font_size: int = Field(
        default=12,
        ge=8,
        le=24,
        description="Default font size for plots"
    )


class ProcessingConfig(BaseModel):
    """Configuration for data processing parameters."""
    
    max_workers: Optional[int] = Field(
        default=None,
        description="Maximum number of worker processes (None = auto-detect)"
    )
    memory_limit_gb: float = Field(
        default=8.0,
        gt=0,
        description="Memory limit in GB for processing"
    )
    temp_dir: Optional[Path] = Field(
        default=None,
        description="Temporary directory for processing"
    )
    
    @field_validator('temp_dir')
    @classmethod
    def validate_temp_dir(cls, v):
        """Ensure temp directory exists or can be created."""
        if v is not None:
            v = Path(v)
            if not v.exists():
                try:
                    v.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    raise ValueError(f"Cannot create temp directory {v}: {e}")
        return v


class CalculationConfig(BaseModel):
    """Configuration for forest metric calculations."""
    
    name: str = Field(description="Name of the calculation")
    enabled: bool = Field(default=True, description="Whether this calculation is enabled")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Calculation-specific parameters"
    )
    output_format: str = Field(
        default="geotiff",
        description="Output format: 'geotiff', 'zarr', 'netcdf'"
    )
    output_name: Optional[str] = Field(
        default=None,
        description="Custom output filename (if None, uses calculation name)"
    )




class GridFIASettings(BaseSettings):
    """Main settings class for GridFIA application."""

    # Application info
    app_name: str = "GridFIA"
    debug: bool = Field(default=False, description="Enable debug mode")
    verbose: bool = Field(default=False, description="Enable verbose output")
    
    # File paths
    data_dir: Path = Field(
        default=Path("data"),
        description="Base directory for data files"
    )
    output_dir: Path = Field(
        default=Path("output"),
        description="Base directory for output files"
    )
    cache_dir: Path = Field(
        default=Path(".cache"),
        description="Directory for caching intermediate results"
    )
    
    # Processing configurations  
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    
    # Calculation configurations
    calculations: List[CalculationConfig] = Field(
        default_factory=lambda: [
            CalculationConfig(
                name="species_richness",
                parameters={"biomass_threshold": 0.0}
            ),
            CalculationConfig(
                name="total_biomass",
                enabled=False
            ),
            CalculationConfig(
                name="shannon_diversity",
                enabled=False
            )
        ],
        description="List of calculations to perform"
    )
    
    # Data validation
    species_codes: List[str] = Field(
        default_factory=list,
        description="List of valid species codes"
    )
    
    model_config = ConfigDict(
        env_prefix="GRIDFIA_",     # Environment variables start with GRIDFIA_
        env_file=".env",           # Load from .env file if present
        case_sensitive=False,      # Case-insensitive environment variables
        extra="ignore"             # Ignore extra fields in config files
    )
    
    @field_validator('data_dir', 'output_dir', 'cache_dir')
    @classmethod
    def ensure_directories_exist(cls, v):
        """Ensure directories exist."""
        v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    def get_output_path(self, filename: str) -> Path:
        """Get full output path for a filename."""
        return self.output_dir / filename
    
    def get_temp_path(self, filename: str) -> Path:
        """Get temporary file path."""
        temp_dir = self.processing.temp_dir or self.cache_dir
        return temp_dir / filename


# Global settings instance
settings = GridFIASettings()

# Backwards compatibility alias (deprecated)
BigMapSettings = GridFIASettings


def load_settings(config_file: Optional[Path] = None) -> GridFIASettings:
    """
    Load settings from file or environment.
    
    Args:
        config_file: Optional path to configuration file
        
    Returns:
        Configured settings instance
    """
    if config_file and config_file.exists():
        # Load from JSON/YAML file
        import json
        import yaml
        
        config_file = Path(config_file)
        
        with open(config_file) as f:
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                config_data = yaml.safe_load(f)
            else:
                config_data = json.load(f)
        return GridFIASettings(**config_data)
    else:
        # Load from environment/defaults
        return GridFIASettings()


def save_settings(settings_obj: GridFIASettings, config_file: Path) -> None:
    """
    Save settings to file.
    
    Args:
        settings_obj: Settings to save
        config_file: Path to save configuration
    """
    import json
    
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'w') as f:
        json.dump(
            settings_obj.model_dump(),
            f,
            indent=2,
            default=str  # Handle Path objects
        ) 