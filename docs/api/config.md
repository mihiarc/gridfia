# Configuration API Reference

The configuration system uses Pydantic v2 for type-safe settings management.

## BigMapSettings

Main settings class for BigMap application.

```python
class BigMapSettings(BaseSettings):
    """Main settings class for BigMap application."""
```

### Attributes

#### Application Settings
- `app_name` (str): Application name (default: "BigMap")
- `debug` (bool): Enable debug mode (default: False)
- `verbose` (bool): Enable verbose output (default: False)

#### Directory Settings
- `data_dir` (Path): Base directory for data files (default: "data")
- `output_dir` (Path): Base directory for output files (default: "output")
- `cache_dir` (Path): Directory for caching intermediate results (default: ".cache")

#### Processing Configuration
- `visualization` (VisualizationConfig): Visualization parameters
- `processing` (ProcessingConfig): Data processing parameters

#### Calculation Configuration
- `calculations` (List[CalculationConfig]): List of calculations to perform
- `species_codes` (List[str]): List of valid species codes

### Environment Variables

Settings can be configured via environment variables with the `BIGMAP_` prefix:

```bash
export BIGMAP_DEBUG=true
export BIGMAP_OUTPUT_DIR=/path/to/output
export BIGMAP_DATA_DIR=/path/to/data
```

### Configuration File

Settings can be loaded from JSON or YAML files:

```yaml
# config.yaml
app_name: BigMap Analysis
debug: false
output_dir: results/
calculations:
  - name: species_richness
    enabled: true
    parameters:
      biomass_threshold: 0.5
  - name: shannon_diversity
    enabled: true
    output_format: netcdf
```

## Configuration Classes

### CalculationConfig

Configuration for individual calculations.

```python
class CalculationConfig(BaseModel):
    """Configuration for forest metric calculations."""
    
    name: str  # Name of the calculation
    enabled: bool = True  # Whether this calculation is enabled
    parameters: Dict[str, Any] = {}  # Calculation-specific parameters
    output_format: str = "geotiff"  # Output format
    output_name: Optional[str] = None  # Custom output filename
```

### ProcessingConfig

Configuration for data processing.

```python
class ProcessingConfig(BaseModel):
    """Configuration for data processing parameters."""
    
    max_workers: Optional[int] = None  # Max worker processes
    memory_limit_gb: float = 8.0  # Memory limit in GB
    temp_dir: Optional[Path] = None  # Temporary directory
```

### VisualizationConfig

Configuration for visualization.

```python
class VisualizationConfig(BaseModel):
    """Configuration for visualization parameters."""
    
    default_dpi: int = 300  # Default DPI for output images
    default_figure_size: Tuple[float, float] = (16, 12)  # Figure size
    color_maps: dict = {...}  # Default color maps
    font_size: int = 12  # Default font size
```

## Helper Functions

### load_settings

```python
load_settings(config_file: Optional[Path] = None) -> BigMapSettings
```

Load settings from file or environment.

**Example:**
```python
from bigmap.config import load_settings

# Load from file
settings = load_settings("config.yaml")

# Load from environment/defaults
settings = load_settings()
```

### save_settings

```python
save_settings(settings_obj: BigMapSettings, config_file: Path) -> None
```

Save settings to file.

**Example:**
```python
from bigmap.config import BigMapSettings, save_settings

settings = BigMapSettings(
    output_dir="results/",
    calculations=[...]
)
save_settings(settings, "my_config.json")
```

## Usage Examples

### Basic Configuration

```python
from bigmap.config import BigMapSettings, CalculationConfig

settings = BigMapSettings(
    output_dir="analysis_results",
    calculations=[
        CalculationConfig(
            name="species_richness",
            enabled=True,
            parameters={"biomass_threshold": 1.0}
        ),
        CalculationConfig(
            name="total_biomass",
            enabled=True,
            output_format="zarr"
        )
    ]
)
```

### Loading from YAML

```python
from pathlib import Path
from bigmap.config import load_settings

# Create config file
config_yaml = """
output_dir: results/diversity_analysis
calculations:
  - name: species_richness
    enabled: true
  - name: shannon_diversity
    enabled: true
    output_format: netcdf
"""

Path("config.yaml").write_text(config_yaml)

# Load settings
settings = load_settings("config.yaml")
```

### Programmatic Configuration

```python
from bigmap.config import BigMapSettings, CalculationConfig

# Create settings programmatically
settings = BigMapSettings()

# Add calculations dynamically
settings.calculations.append(
    CalculationConfig(
        name="dominant_species",
        enabled=True,
        output_name="dominant_species_map"
    )
)

# Update processing configuration
settings.processing.memory_limit_gb = 16.0
settings.processing.max_workers = 8
```

## Best Practices

1. **Use Configuration Files**: Store complex configurations in YAML/JSON files
2. **Environment Variables**: Use for deployment-specific settings (paths, debug flags)
3. **Validation**: Pydantic automatically validates all settings
4. **Type Safety**: Use type hints for all configuration parameters
5. **Defaults**: Provide sensible defaults for all optional parameters