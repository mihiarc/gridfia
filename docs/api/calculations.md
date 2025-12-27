# Calculations

GridFIA provides 15+ forest metrics through a flexible plugin-based calculation framework.
Calculations are registered with a central registry and can be executed via the `GridFIA.calculate_metrics()` method.

## Overview

### Diversity Metrics

| Calculation | Description | Units |
|-------------|-------------|-------|
| `species_richness` | Count of species per pixel | count |
| `shannon_diversity` | Shannon diversity index (H') | nats or bits |
| `simpson_diversity` | Simpson diversity index (1-D) | unitless |
| `evenness` | Pielou's evenness (J) | unitless |

### Biomass Metrics

| Calculation | Description | Units |
|-------------|-------------|-------|
| `total_biomass` | Sum of biomass across all species | Mg/ha |
| `species_proportion` | Species as proportion of total | ratio |
| `species_percentage` | Species as percentage of total | percent |
| `species_group_proportion` | Group-level proportions | ratio |
| `biomass_threshold` | Pixels exceeding threshold | binary |

### Species Analysis

| Calculation | Description | Units |
|-------------|-------------|-------|
| `dominant_species` | Most abundant species per pixel | species index |
| `species_presence` | Presence/absence mapping | binary |
| `species_dominance` | Dominance indices | unitless |
| `rare_species` | Species with low biomass | count |
| `common_species` | Widely distributed species | count |

## Quick Start

```python
from gridfia import GridFIA

api = GridFIA()

# List all available calculations
calcs = api.list_calculations()
print(f"Available: {calcs}")

# Run specific calculations
results = api.calculate_metrics(
    "data/forest.zarr",
    calculations=["species_richness", "shannon_diversity", "total_biomass"]
)

for result in results:
    print(f"{result.name}: {result.output_path}")
```

## Using the Registry Directly

For advanced use cases, access the calculation registry directly:

```python
from gridfia.core.calculations import registry
import numpy as np

# List all calculations
calcs = registry.list_calculations()

# Get a calculation instance with parameters
calc = registry.get('species_richness', biomass_threshold=1.0)

# Run on data (3D array: species x height x width)
biomass_data = np.random.rand(10, 100, 100) * 50
result = calc.calculate(biomass_data)

print(f"Richness range: {result.min()} - {result.max()}")
```

## Calculation Reference

### species_richness

Count of species with biomass above threshold at each pixel.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `biomass_threshold` | `float` | `0.0` | Minimum biomass to count species |
| `exclude_total_layer` | `bool` | `True` | Exclude pre-calculated total layer |

**Example:**

```python
from gridfia import GridFIA

api = GridFIA()
results = api.calculate_metrics(
    "data.zarr",
    calculations=["species_richness"]
)

# With custom threshold via config
from gridfia.config import CalculationConfig, GridFIASettings

settings = GridFIASettings(
    calculations=[
        CalculationConfig(
            name="species_richness",
            parameters={"biomass_threshold": 0.5}
        )
    ]
)
api = GridFIA(config=settings)
results = api.calculate_metrics("data.zarr")
```

### shannon_diversity

Shannon diversity index (H') measuring species diversity.

$$H' = -\sum_{i=1}^{S} p_i \ln(p_i)$$

Where $p_i$ is the proportion of species $i$.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base` | `str` | `'e'` | Logarithm base ('e', '2', or '10') |
| `exclude_total_layer` | `bool` | `True` | Exclude pre-calculated total |

**Example:**

```python
results = api.calculate_metrics(
    "data.zarr",
    calculations=["shannon_diversity"]
)
```

### simpson_diversity

Simpson diversity index (1-D) measuring the probability that two randomly selected individuals belong to different species.

$$1 - D = 1 - \sum_{i=1}^{S} p_i^2$$

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `exclude_total_layer` | `bool` | `True` | Exclude pre-calculated total |

**Example:**

```python
results = api.calculate_metrics(
    "data.zarr",
    calculations=["simpson_diversity"]
)
```

### evenness

Pielou's evenness index (J) measuring how evenly species abundances are distributed.

$$J = \frac{H'}{\ln(S)}$$

Where $H'$ is Shannon diversity and $S$ is species richness.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `exclude_total_layer` | `bool` | `True` | Exclude pre-calculated total |

### total_biomass

Sum of biomass across all species at each pixel.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `exclude_total_layer` | `bool` | `True` | Calculate from species (vs. use pre-calculated) |

**Example:**

```python
results = api.calculate_metrics(
    "data.zarr",
    calculations=["total_biomass"]
)
```

### dominant_species

Index of the species with highest biomass at each pixel.

**Output:** Integer array with species indices

**Example:**

```python
from gridfia import GridFIA
from gridfia.utils.zarr_utils import ZarrStore

api = GridFIA()
results = api.calculate_metrics(
    "data.zarr",
    calculations=["dominant_species"]
)

# Map indices to species names
with ZarrStore.open("data.zarr") as store:
    species_names = store.species_names
    # dominant_index = result array
    # species_names[dominant_index] gives species name
```

### species_presence

Binary presence/absence for specific species.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `species_index` | `int` | (required) | Index of species to check |
| `biomass_threshold` | `float` | `0.0` | Minimum biomass for presence |

## Custom Calculations

Create custom calculations by inheriting from `ForestCalculation`:

```python
import numpy as np
from gridfia.core.calculations.base import ForestCalculation
from gridfia.core.calculations.registry import registry

class BiomassDensityRatio(ForestCalculation):
    """Calculate ratio of hardwood to softwood biomass."""

    def __init__(self, hardwood_indices: list = None, softwood_indices: list = None):
        super().__init__(
            name="biomass_density_ratio",
            description="Ratio of hardwood to softwood biomass",
            units="ratio"
        )
        self.hardwood_indices = hardwood_indices or []
        self.softwood_indices = softwood_indices or []

    def calculate(self, biomass_data: np.ndarray, **kwargs) -> np.ndarray:
        """Calculate hardwood/softwood ratio."""
        hardwood = biomass_data[self.hardwood_indices].sum(axis=0)
        softwood = biomass_data[self.softwood_indices].sum(axis=0)

        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(softwood > 0, hardwood / softwood, np.nan)

        return ratio

    def validate_data(self, biomass_data: np.ndarray) -> bool:
        """Validate input data."""
        if biomass_data.ndim != 3:
            return False
        if max(self.hardwood_indices + self.softwood_indices) >= biomass_data.shape[0]:
            return False
        return True

# Register the custom calculation
@registry.register("biomass_density_ratio")
def create_biomass_ratio(**kwargs):
    return BiomassDensityRatio(**kwargs)
```

## Base Class Reference

::: gridfia.core.calculations.base.ForestCalculation
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - __init__
        - calculate
        - validate_data
        - get_output_dtype
        - preprocess_data
        - postprocess_result

## Registry Reference

::: gridfia.core.calculations.registry.CalculationRegistry
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - register
        - get
        - list_calculations
        - get_calculation_info

## Performance Tips

### Memory Management

For large datasets, calculations process data in chunks:

```python
from gridfia import GridFIA, GridFIASettings
from gridfia.config import ProcessingConfig

settings = GridFIASettings(
    processing=ProcessingConfig(
        memory_limit_gb=16.0,
        max_workers=4
    )
)

api = GridFIA(config=settings)
results = api.calculate_metrics("large_data.zarr")
```

### Output Formats

Choose output format based on use case:

| Format | Best For | Size |
|--------|----------|------|
| GeoTIFF | GIS software, QGIS | Compressed |
| Zarr | Large outputs, cloud | Chunked |
| NetCDF | xarray, scientific | Compressed |

```python
from gridfia.config import CalculationConfig, OutputFormat

calc = CalculationConfig(
    name="total_biomass",
    output_format=OutputFormat.ZARR  # For large outputs
)
```

### Parallel Processing

Multiple calculations run in parallel when possible:

```python
# Run multiple calculations efficiently
results = api.calculate_metrics(
    "data.zarr",
    calculations=[
        "species_richness",
        "shannon_diversity",
        "simpson_diversity",
        "evenness",
        "total_biomass"
    ]
)
```

## See Also

- [GridFIA Class](gridfia.md) - Main API for running calculations
- [Configuration](config.md) - Calculation configuration
- [Utilities](utilities.md) - Working with Zarr stores
