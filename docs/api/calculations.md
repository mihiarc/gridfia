# Calculations API Reference

The calculations module provides a flexible framework for implementing forest metric calculations.

## Base Class

### ForestCalculation

Abstract base class for all forest calculations.

```python
class ForestCalculation(ABC):
    """Abstract base class for forest calculations."""
    
    def __init__(self, name: str, description: str, units: str, **kwargs):
        """
        Initialize a forest calculation.
        
        Parameters
        ----------
        name : str
            Unique name for the calculation
        description : str
            Human-readable description
        units : str
            Units of the calculated metric
        **kwargs : dict
            Additional configuration parameters
        """
```

#### Abstract Methods

Every calculation must implement:

1. **calculate**
   ```python
   calculate(biomass_data: np.ndarray, **kwargs) -> np.ndarray
   ```
   Calculate metric from biomass data.

2. **validate_data**
   ```python
   validate_data(biomass_data: np.ndarray) -> bool
   ```
   Validate input data for this calculation.

#### Optional Methods

- `get_output_dtype()`: Specify output data type (default: float32)
- `preprocess_data()`: Custom preprocessing
- `postprocess_result()`: Custom postprocessing

## Available Calculations

### Diversity Metrics

#### SpeciesRichness
Count of species with biomass above threshold.

```python
SpeciesRichness(biomass_threshold: float = 0.0, exclude_total_layer: bool = True)
```

**Parameters:**
- `biomass_threshold`: Minimum biomass to count species as present
- `exclude_total_layer`: Whether to exclude first layer (pre-calculated total)

#### ShannonDiversity
Shannon diversity index (H').

```python
ShannonDiversity(base: str = 'e', exclude_total_layer: bool = True)
```

**Parameters:**
- `base`: Logarithm base ('e' for natural, '2' for bits, '10' for dits)

#### SimpsonDiversity
Simpson diversity index (1 - D).

```python
SimpsonDiversity(exclude_total_layer: bool = True)
```

### Biomass Metrics

#### TotalBiomass
Sum of biomass across all species.

```python
TotalBiomass(exclude_total_layer: bool = True)
```

#### SpeciesProportion
Proportion of specific species relative to total.

```python
SpeciesProportion(species_indices: List[int], exclude_total_layer: bool = True)
```

**Parameters:**
- `species_indices`: List of species layer indices to calculate proportion for

### Species Analysis

#### DominantSpecies
Identify the most abundant species by biomass.

```python
DominantSpecies(exclude_total_layer: bool = True)
```

**Returns:** Species index (int16)

#### SpeciesPresence
Binary presence/absence of specific species.

```python
SpeciesPresence(species_index: int, biomass_threshold: float = 0.0)
```

## Registry System

### Using the Registry

```python
from bigmap.core.calculations import registry

# List available calculations
calculations = registry.list_calculations()
# ['species_richness', 'shannon_diversity', ...]

# Get calculation instance
calc = registry.get('species_richness', biomass_threshold=1.0)

# Get calculation info
info = registry.get_calculation_info('species_richness')
# {'name': 'species_richness', 'description': '...', 'units': 'count', ...}
```

### Registering Custom Calculations

```python
from bigmap.core.calculations import ForestCalculation, register_calculation

class CustomMetric(ForestCalculation):
    def __init__(self):
        super().__init__(
            name="custom_metric",
            description="My custom forest metric",
            units="custom_units"
        )
    
    def calculate(self, biomass_data, **kwargs):
        # Implementation
        return result
    
    def validate_data(self, biomass_data):
        return biomass_data.ndim == 3

# Register the calculation
register_calculation("custom_metric", CustomMetric)
```

## Example Usage

```python
import numpy as np
from bigmap.core.calculations import registry

# Get sample data (species, y, x)
biomass_data = np.random.rand(5, 100, 100) * 100

# Calculate species richness
richness_calc = registry.get('species_richness', biomass_threshold=0.5)
richness = richness_calc.calculate(biomass_data)

# Calculate Shannon diversity
shannon_calc = registry.get('shannon_diversity')
diversity = shannon_calc.calculate(biomass_data)

# Get dominant species
dominant_calc = registry.get('dominant_species')
dominant = dominant_calc.calculate(biomass_data)
```

## Performance Tips

1. **Chunking**: Process large arrays in chunks to manage memory
2. **Data Types**: Use appropriate output dtypes (uint8 for counts, float32 for ratios)
3. **Validation**: Implement efficient validation to fail fast on invalid data
4. **Preprocessing**: Cache preprocessing results when possible