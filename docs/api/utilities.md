# Utilities

GridFIA provides utility classes for advanced users who need direct access to
Zarr stores and location configuration.

## Overview

| Class | Description | Use Case |
|-------|-------------|----------|
| [`ZarrStore`](#zarrstore) | Unified Zarr access interface | Reading biomass data directly |
| [`LocationConfig`](#locationconfig) | Geographic extent management | Custom region configuration |

## ZarrStore

The `ZarrStore` class provides a unified interface for reading GridFIA Zarr stores,
handling both Zarr v2 and v3 formats transparently.

::: gridfia.utils.zarr_utils.ZarrStore
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - from_path
        - open
        - is_valid_store
        - close
        - biomass
        - species_codes
        - species_names
        - crs
        - transform
        - bounds
        - num_species
        - shape

### Basic Usage

```python
from gridfia.utils.zarr_utils import ZarrStore

# Open a Zarr store
store = ZarrStore.from_path("data/forest.zarr")

# Access properties
print(f"Shape: {store.shape}")  # (species, height, width)
print(f"Species: {store.num_species}")
print(f"CRS: {store.crs}")
print(f"Bounds: {store.bounds}")

# Access species information
for code, name in zip(store.species_codes, store.species_names):
    print(f"  {code}: {name}")

# Access biomass data
biomass = store.biomass[:]  # Load all data
print(f"Biomass shape: {biomass.shape}")
print(f"Total biomass range: {biomass[0].min():.2f} - {biomass[0].max():.2f}")

# Close when done
store.close()
```

### Context Manager Usage

Use the context manager for automatic resource cleanup:

```python
from gridfia.utils.zarr_utils import ZarrStore

with ZarrStore.open("data/forest.zarr") as store:
    # Access data within context
    biomass = store.biomass[:]
    species = store.species_codes

    # Process data
    total_biomass = biomass[0]  # First layer is total
    mean_biomass = total_biomass[total_biomass > 0].mean()
    print(f"Mean biomass: {mean_biomass:.2f} Mg/ha")
# Store automatically closed
```

### Working with Species Data

```python
from gridfia.utils.zarr_utils import ZarrStore
import numpy as np

with ZarrStore.open("data/forest.zarr") as store:
    # Get index of specific species
    species_code = "0202"  # Douglas-fir
    try:
        idx = store.species_codes.index(species_code)
        species_data = store.biomass[idx]

        # Calculate statistics
        valid_data = species_data[species_data > 0]
        print(f"Species {species_code}:")
        print(f"  Mean biomass: {valid_data.mean():.2f} Mg/ha")
        print(f"  Max biomass: {valid_data.max():.2f} Mg/ha")
        print(f"  Coverage: {len(valid_data) / species_data.size * 100:.1f}%")

    except ValueError:
        print(f"Species {species_code} not in store")
```

### Chunked Reading for Large Datasets

```python
from gridfia.utils.zarr_utils import ZarrStore
import numpy as np

with ZarrStore.open("data/large_forest.zarr") as store:
    # Read in chunks to avoid memory issues
    chunk_size = 1000
    height, width = store.shape[1], store.shape[2]

    total_sum = 0
    total_count = 0

    for y in range(0, height, chunk_size):
        for x in range(0, width, chunk_size):
            y_end = min(y + chunk_size, height)
            x_end = min(x + chunk_size, width)

            # Read chunk
            chunk = store.biomass[0, y:y_end, x:x_end]
            valid = chunk[chunk > 0]

            total_sum += valid.sum()
            total_count += len(valid)

    mean_biomass = total_sum / total_count if total_count > 0 else 0
    print(f"Mean biomass: {mean_biomass:.2f} Mg/ha")
```

### Validation

```python
from gridfia.utils.zarr_utils import ZarrStore
from pathlib import Path

# Quick validation without full loading
path = Path("data/forest.zarr")
if ZarrStore.is_valid_store(path):
    print("Valid GridFIA Zarr store")
else:
    print("Invalid or non-GridFIA Zarr store")
```

## LocationConfig

The `LocationConfig` class manages geographic extents for any US state, county, or custom region.

::: gridfia.utils.location_config.LocationConfig
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - from_state
        - from_county
        - from_bbox

### Creating Location Configurations

=== "From State"

    ```python
    from gridfia.utils.location_config import LocationConfig

    # Create configuration for a state
    config = LocationConfig.from_state("Montana")

    print(f"Location: {config.location_name}")
    print(f"Bbox (Web Mercator): {config.web_mercator_bbox}")

    # Save to file for reuse
    config = LocationConfig.from_state(
        "Montana",
        output_path="config/montana.yaml"
    )
    ```

=== "From County"

    ```python
    from gridfia.utils.location_config import LocationConfig

    # Create configuration for a county
    config = LocationConfig.from_county(
        county="Wake",
        state="North Carolina"
    )

    print(f"Location: {config.location_name}")
    print(f"Bbox: {config.web_mercator_bbox}")

    # Save configuration
    config = LocationConfig.from_county(
        county="Harris",
        state="Texas",
        output_path="config/harris_county.yaml"
    )
    ```

=== "From Bounding Box"

    ```python
    from gridfia.utils.location_config import LocationConfig

    # WGS84 bounding box (lon/lat)
    config = LocationConfig.from_bbox(
        bbox=(-123.5, 45.0, -122.0, 46.5),
        name="Pacific Northwest Study Area",
        crs="EPSG:4326"
    )

    print(f"Location: {config.location_name}")
    print(f"Web Mercator bbox: {config.web_mercator_bbox}")

    # Save custom region
    config = LocationConfig.from_bbox(
        bbox=(-123.5, 45.0, -122.0, 46.5),
        name="PNW Study Area",
        crs="EPSG:4326",
        output_path="config/pnw_study.yaml"
    )
    ```

### Using Configurations

```python
from gridfia import GridFIA
from gridfia.utils.location_config import LocationConfig
from pathlib import Path

api = GridFIA()

# Create and save configuration
config = LocationConfig.from_county(
    county="Wake",
    state="NC",
    output_path="config/wake.yaml"
)

# Use configuration for download
files = api.download_species(
    location_config="config/wake.yaml",
    species_codes=["0131", "0316"],
    output_dir="data/wake"
)
```

### Loading Saved Configurations

```python
from gridfia.utils.location_config import LocationConfig
from pathlib import Path

# Load from file
config = LocationConfig(Path("config/montana.yaml"))

print(f"Location: {config.location_name}")
print(f"Bbox: {config.web_mercator_bbox}")
```

## Helper Functions

### Zarr Creation

```python
from gridfia.utils.zarr_utils import create_zarr_from_geotiffs, validate_zarr_store
from pathlib import Path

# Create Zarr from GeoTIFFs
create_zarr_from_geotiffs(
    output_zarr_path=Path("data/forest.zarr"),
    geotiff_paths=[
        Path("downloads/species_0202.tif"),
        Path("downloads/species_0122.tif"),
    ],
    species_codes=["0202", "0122"],
    species_names=["Douglas-fir", "Ponderosa pine"],
    chunk_size=(1, 1000, 1000),
    compression="lz4",
    compression_level=5,
    include_total=True
)

# Validate the created store
info = validate_zarr_store(Path("data/forest.zarr"))
print(f"Shape: {info['shape']}")
print(f"Species: {info['num_species']}")
print(f"Chunks: {info['chunks']}")
print(f"CRS: {info['crs']}")
```

### Configuration Loading

```python
from gridfia.config import load_settings, save_settings, GridFIASettings

# Load from file
settings = load_settings(Path("config/production.yaml"))

# Save current settings
save_settings(settings, Path("config/backup.json"))
```

## Integration with NumPy and Xarray

### NumPy Integration

```python
from gridfia.utils.zarr_utils import ZarrStore
import numpy as np

with ZarrStore.open("data/forest.zarr") as store:
    # Load as NumPy array
    biomass = np.asarray(store.biomass)

    # Calculate species richness
    presence = biomass > 0
    richness = presence.sum(axis=0)

    print(f"Max richness: {richness.max()} species")
```

### Xarray Integration

```python
from gridfia.utils.zarr_utils import ZarrStore
import xarray as xr
import numpy as np

with ZarrStore.open("data/forest.zarr") as store:
    # Create xarray DataArray
    da = xr.DataArray(
        store.biomass[:],
        dims=["species", "y", "x"],
        coords={
            "species": store.species_codes,
        },
        attrs={
            "crs": str(store.crs),
            "units": "Mg/ha"
        }
    )

    # Xarray operations
    total = da.sum(dim="species")
    mean_by_species = da.mean(dim=["y", "x"])

    print(mean_by_species)
```

## See Also

- [GridFIA Class](gridfia.md) - High-level API
- [Data Models](models.md) - API return types
- [Configuration](config.md) - Settings management
