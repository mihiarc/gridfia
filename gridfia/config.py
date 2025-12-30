"""
Configuration management for GridFIA using Pydantic.

This module defines configuration schemas and settings management
for the GridFIA package, part of the FIA Python Ecosystem.
"""

from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from pydantic import BaseModel, Field, field_validator, field_serializer
from pydantic_settings import BaseSettings, SettingsConfigDict


class OutputFormat(str, Enum):
    """Supported output formats for calculation results."""
    GEOTIFF = "geotiff"
    ZARR = "zarr"
    NETCDF = "netcdf"


class CloudStorageBackend(str, Enum):
    """Supported cloud storage backends."""
    BACKBLAZE_B2 = "b2"
    CLOUDFLARE_R2 = "r2"
    AWS_S3 = "s3"
    HTTP = "http"  # Generic HTTP/HTTPS URLs


class CloudStorageConfig(BaseModel):
    """Configuration for cloud storage of Zarr datasets.

    Supports multiple backends for storing and streaming forest data:
    - Backblaze B2: Low-cost S3-compatible storage
    - Cloudflare R2: Zero egress fees
    - AWS S3: Standard cloud storage
    - HTTP: Any publicly accessible HTTP URLs

    Environment variables (with GRIDFIA_ prefix):
        GRIDFIA_CLOUD_BACKEND: Storage backend (b2, r2, s3, http)
        GRIDFIA_CLOUD_BUCKET: Bucket name
        GRIDFIA_CLOUD_PUBLIC_URL: Public URL for data access
        GRIDFIA_CLOUD_ENDPOINT_URL: S3-compatible endpoint (for b2/r2)
        GRIDFIA_CLOUD_ACCESS_KEY: Access key (optional, for private buckets)
        GRIDFIA_CLOUD_SECRET_KEY: Secret key (optional, for private buckets)
    """

    backend: CloudStorageBackend = Field(
        default=CloudStorageBackend.BACKBLAZE_B2,
        description="Cloud storage backend type"
    )
    bucket: str = Field(
        default="gridfia-data",
        description="Bucket name for storing data"
    )
    public_url: str = Field(
        default="https://f004.backblazeb2.com/file/gridfia-data",
        description="Public URL base for accessing data (no trailing slash)"
    )
    endpoint_url: Optional[str] = Field(
        default=None,
        description="S3-compatible endpoint URL (for B2/R2). "
                    "B2: https://s3.us-west-004.backblazeb2.com, "
                    "R2: https://<account>.r2.cloudflarestorage.com"
    )
    region: str = Field(
        default="us-west-004",
        description="Cloud region (for S3-compatible backends)"
    )
    access_key: Optional[str] = Field(
        default=None,
        description="Access key for authenticated access (optional for public buckets)"
    )
    secret_key: Optional[str] = Field(
        default=None,
        description="Secret key for authenticated access (optional for public buckets)"
    )

    # Path prefixes within the bucket
    states_prefix: str = Field(
        default="states",
        description="Path prefix for state datasets (e.g., states/ri/ri_forest.zarr)"
    )
    samples_prefix: str = Field(
        default="samples",
        description="Path prefix for sample datasets (e.g., samples/durham_nc.zarr)"
    )

    def get_state_url(self, state_abbr: str) -> str:
        """Get the public URL for a state's Zarr dataset.

        Args:
            state_abbr: Two-letter state abbreviation (e.g., 'RI', 'NC')

        Returns:
            Full URL to the state's Zarr store
        """
        state_lower = state_abbr.lower()
        return f"{self.public_url}/{self.states_prefix}/{state_lower}/{state_lower}_forest.zarr"

    def get_sample_url(self, sample_name: str) -> str:
        """Get the public URL for a sample dataset.

        Args:
            sample_name: Name of the sample (e.g., 'durham_nc')

        Returns:
            Full URL to the sample Zarr store
        """
        return f"{self.public_url}/{self.samples_prefix}/{sample_name}.zarr"

    def get_storage_options(self, url: Optional[str] = None) -> Dict[str, Any]:
        """Get fsspec storage options for this backend.

        Args:
            url: Optional URL to determine protocol. If starts with http(s),
                 returns empty options for public HTTP access.

        Returns:
            Dictionary of options to pass to fsspec/zarr for cloud access
        """
        options: Dict[str, Any] = {}

        # For HTTP/HTTPS URLs (public bucket access), no special options needed
        # fsspec uses aiohttp backend which doesn't need S3 credentials
        if url and url.startswith(("http://", "https://")):
            return options

        if self.backend == CloudStorageBackend.HTTP:
            # HTTP backend explicitly set - no special options
            return options

        # S3-compatible backends (B2, R2, S3) - only for s3:// URLs
        if self.access_key and self.secret_key:
            options["key"] = self.access_key
            options["secret"] = self.secret_key
        else:
            # Anonymous access for public buckets
            options["anon"] = True

        if self.endpoint_url:
            options["client_kwargs"] = {"endpoint_url": self.endpoint_url}

        if self.region:
            options["client_kwargs"] = options.get("client_kwargs", {})
            options["client_kwargs"]["region_name"] = self.region

        return options

    def get_s3_url(self, path: str) -> str:
        """Get S3-style URL for a path (for authenticated access).

        Args:
            path: Path within the bucket (e.g., 'states/ri/ri_forest.zarr')

        Returns:
            S3-style URL (s3://bucket/path)
        """
        return f"s3://{self.bucket}/{path}"


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
    color_maps: Dict[str, str] = Field(
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

    name: str = Field(min_length=1, description="Name of the calculation")
    enabled: bool = Field(default=True, description="Whether this calculation is enabled")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Calculation-specific parameters"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.GEOTIFF,
        description="Output format for calculation results"
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

    # Cloud storage configuration
    cloud: CloudStorageConfig = Field(
        default_factory=CloudStorageConfig,
        description="Cloud storage settings for remote Zarr access"
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
        min_length=1,
        description="List of calculations to perform (must not be empty)"
    )
    
    # Data validation
    species_codes: List[str] = Field(
        default_factory=list,
        description="List of valid species codes"
    )
    
    model_config = SettingsConfigDict(
        env_prefix="GRIDFIA_",           # Environment variables start with GRIDFIA_
        env_file=".env",                 # Load from .env file if present
        env_nested_delimiter="__",       # Use __ for nested config (e.g., GRIDFIA_CLOUD__BUCKET)
        case_sensitive=False,            # Case-insensitive environment variables
        extra="ignore"                   # Ignore extra fields in config files
    )
    
    @field_validator('data_dir', 'output_dir', 'cache_dir')
    @classmethod
    def ensure_directories_exist(cls, v):
        """Ensure directories exist."""
        v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v

    @field_serializer('data_dir', 'output_dir', 'cache_dir')
    def serialize_path(self, v: Path) -> str:
        """Serialize Path objects to strings for JSON compatibility."""
        return str(v)

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