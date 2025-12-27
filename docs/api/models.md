# Data Models

GridFIA uses Pydantic models for type-safe data validation and serialization.
These models represent the data structures returned by API methods.

## Overview

| Model | Description | Used By |
|-------|-------------|---------|
| [`SpeciesInfo`](#speciesinfo) | Tree species metadata | `GridFIA.list_species()` |
| [`CalculationResult`](#calculationresult) | Calculation output | `GridFIA.calculate_metrics()` |

## SpeciesInfo

Information about a tree species from the FIA BIGMAP service.

::: gridfia.api.SpeciesInfo
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

### Example Usage

```python
from gridfia import GridFIA

api = GridFIA()
species = api.list_species()

# Access species information
for sp in species[:5]:
    print(f"{sp.species_code}: {sp.common_name}")
    print(f"  Scientific: {sp.scientific_name}")
    if sp.function_name:
        print(f"  Function: {sp.function_name}")

# Filter species
oaks = [s for s in species if "oak" in s.common_name.lower()]
print(f"Found {len(oaks)} oak species")

# Convert to dictionary
species_dict = species[0].model_dump()
print(species_dict)
# {'species_code': '0131', 'common_name': 'Loblolly pine',
#  'scientific_name': 'Pinus taeda', 'function_name': 'SP0131_LoblollyPine'}
```

### Species Codes

FIA species codes are 4-digit identifiers. Common examples:

| Code | Common Name | Scientific Name |
|------|-------------|-----------------|
| 0131 | Loblolly pine | *Pinus taeda* |
| 0202 | Douglas-fir | *Pseudotsuga menziesii* |
| 0122 | Ponderosa pine | *Pinus ponderosa* |
| 0068 | Eastern hemlock | *Tsuga canadensis* |
| 0316 | Red maple | *Acer rubrum* |
| 0802 | White oak | *Quercus alba* |
| 0833 | Northern red oak | *Quercus rubra* |
| 0746 | Quaking aspen | *Populus tremuloides* |

The special code `0000` represents **Total Biomass** (sum of all species).

## CalculationResult

Result from a forest metric calculation.

::: gridfia.api.CalculationResult
    options:
      show_root_heading: false
      show_source: false
      heading_level: 3

### Example Usage

```python
from gridfia import GridFIA
from pathlib import Path

api = GridFIA()

# Run calculations
results = api.calculate_metrics(
    "data/forest.zarr",
    calculations=["species_richness", "shannon_diversity", "total_biomass"]
)

# Process results
for result in results:
    print(f"Calculation: {result.name}")
    print(f"  Output: {result.output_path}")

    # Access statistics if available
    if result.statistics:
        for stat, value in result.statistics.items():
            print(f"  {stat}: {value:.4f}")

    # Access metadata
    print(f"  Source: {result.metadata.get('zarr_path')}")

# Filter by name
richness_result = next(r for r in results if r.name == "species_richness")
print(f"Richness output: {richness_result.output_path}")

# Convert to dictionary
result_dict = results[0].model_dump()
```

### Result Structure

Each `CalculationResult` contains:

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Name of the calculation (e.g., "species_richness") |
| `output_path` | `Path` | Absolute path to the output file |
| `statistics` | `dict[str, float]` | Summary statistics (mean, std, min, max) |
| `metadata` | `dict[str, Any]` | Additional metadata (source zarr, parameters) |

## Working with Models

### Serialization

All models support JSON serialization via Pydantic:

```python
import json
from gridfia import GridFIA

api = GridFIA()
species = api.list_species()

# Serialize to JSON
json_str = species[0].model_dump_json(indent=2)
print(json_str)

# Serialize list to JSON
all_species_json = json.dumps(
    [s.model_dump() for s in species],
    indent=2
)

# Write to file
with open("species.json", "w") as f:
    json.dump([s.model_dump() for s in species], f, indent=2)
```

### DataFrame Conversion

Convert to pandas DataFrames for analysis:

```python
import pandas as pd
from gridfia import GridFIA

api = GridFIA()

# Species to DataFrame
species = api.list_species()
species_df = pd.DataFrame([s.model_dump() for s in species])
print(species_df.head())

# Filter and analyze
conifers = species_df[species_df['common_name'].str.contains('pine|fir|spruce', case=False)]
print(f"Found {len(conifers)} conifer species")

# Results to DataFrame
results = api.calculate_metrics("data.zarr", calculations=["species_richness"])
results_df = pd.DataFrame([r.model_dump() for r in results])
```

### Validation

Models validate data on creation:

```python
from gridfia.api import SpeciesInfo
from pydantic import ValidationError

# Valid species
sp = SpeciesInfo(
    species_code="0131",
    common_name="Loblolly pine",
    scientific_name="Pinus taeda"
)

# Invalid species code (must be 4 digits)
try:
    sp = SpeciesInfo(
        species_code="131",  # Missing leading zero
        common_name="Loblolly pine",
        scientific_name="Pinus taeda"
    )
except ValidationError as e:
    print(f"Validation error: {e}")

# Empty name not allowed
try:
    sp = SpeciesInfo(
        species_code="0131",
        common_name="",  # Empty not allowed
        scientific_name="Pinus taeda"
    )
except ValidationError as e:
    print(f"Validation error: {e}")
```

## Type Hints

All models are fully typed for IDE support:

```python
from gridfia import GridFIA
from gridfia.api import SpeciesInfo, CalculationResult
from typing import List

def analyze_species(species: List[SpeciesInfo]) -> dict:
    """Analyze species list with full type hints."""
    return {
        "count": len(species),
        "codes": [s.species_code for s in species],
        "names": [s.common_name for s in species]
    }

def process_results(results: List[CalculationResult]) -> dict:
    """Process calculation results with type hints."""
    return {
        r.name: str(r.output_path)
        for r in results
    }
```

## See Also

- [GridFIA Class](gridfia.md) - Main API interface
- [Configuration](config.md) - Settings and configuration models
- [Exceptions](exceptions.md) - Error handling
