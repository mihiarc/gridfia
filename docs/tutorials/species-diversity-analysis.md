# Tutorial: Species Diversity Analysis

This tutorial demonstrates how to perform a comprehensive species diversity analysis using GridFIA and BIGMAP data.

## Scientific Background

Species diversity is a fundamental measure of ecosystem health and resilience. This tutorial covers three key diversity metrics:

### Shannon Diversity Index (H')

The Shannon diversity index (Shannon, 1948) measures both species richness and evenness:

$$H' = -\sum_{i=1}^{S} p_i \ln(p_i)$$

Where $p_i$ is the proportion of species $i$. Higher values indicate greater diversity.

- Values typically range from 0 to 5
- H' = 0 indicates a monoculture
- H' > 3 indicates high diversity

### Simpson Diversity Index

The Simpson index (Simpson, 1949) has multiple formulations:

**Simpson's Dominance (D)**: $\sum p_i^2$

- Probability that two individuals belong to the same species
- Values range from 0 to 1 (lower = more diverse)

**Simpson's Diversity (1-D)**: $1 - \sum p_i^2$

- Probability that two individuals belong to different species
- Values range from 0 to 1 (higher = more diverse)

GridFIA calculates Simpson's Diversity (1-D) by default.

### Pielou's Evenness (J)

Pielou's evenness (Pielou, 1966) measures how evenly species are distributed:

$$J = \frac{H'}{\ln(S)}$$

Where S is the number of species.

- Values range from 0 to 1
- J = 1 indicates perfect evenness
- J < 0.5 suggests dominance by few species

### When to Use Each Index

| Index | Best For |
|-------|----------|
| **Shannon** | General biodiversity assessment, sensitive to rare species |
| **Simpson** | When dominance patterns are important |
| **Species Richness** | Simple count when presence/absence is sufficient |
| **Evenness** | Assessing community balance independent of richness |

## Overview

We'll analyze forest species diversity by:

1. Downloading species biomass data from BIGMAP
2. Creating a Zarr array for efficient processing
3. Calculating diversity metrics
4. Visualizing and interpreting results

## Prerequisites

- GridFIA installed (`pip install gridfia`)
- Basic Python knowledge
- ~2GB disk space for data

## Step 1: Initialize and Explore

```python
from gridfia import GridFIA

# Initialize the API
api = GridFIA()

# List available species from BIGMAP
species = api.list_species()
print(f"BIGMAP provides data for {len(species)} tree species")

# Display some common species
for s in species[:10]:
    print(f"  {s.species_code}: {s.common_name}")
```

## Step 2: Download Species Data

Download biomass rasters for common North Carolina tree species:

```python
from gridfia import GridFIA

api = GridFIA()

# Define species of interest
species_codes = [
    "0131",  # Loblolly pine
    "0068",  # Eastern white pine
    "0110",  # Shortleaf pine
    "0316",  # Eastern redcedar
    "0611",  # Sweetgum
    "0802",  # White oak
    "0833",  # Northern red oak
]

# Download for Wake County, NC
files = api.download_species(
    state="North Carolina",
    county="Wake",
    species_codes=species_codes,
    output_dir="tutorial_data"
)

print(f"Downloaded {len(files)} species files")
for f in files:
    print(f"  {f}")
```

## Step 3: Create Zarr Store

Convert downloaded GeoTIFF files to cloud-optimized Zarr format:

```python
from gridfia import GridFIA

api = GridFIA()

# Create Zarr store from downloaded rasters
zarr_path = api.create_zarr(
    input_dir="tutorial_data",
    output_path="tutorial_data/wake_forest.zarr",
    chunk_size=(1, 1000, 1000)
)

# Validate the store
info = api.validate_zarr(zarr_path)
print(f"Created Zarr store:")
print(f"  Species: {info['num_species']}")
print(f"  Shape: {info['shape']}")
print(f"  CRS: {info['crs']}")
```

## Step 4: Calculate Diversity Metrics

Run all diversity calculations:

```python
from gridfia import GridFIA

api = GridFIA()

# List available calculations
print("Available calculations:")
for calc in api.list_calculations():
    print(f"  - {calc}")

# Calculate diversity metrics
results = api.calculate_metrics(
    zarr_path="tutorial_data/wake_forest.zarr",
    calculations=[
        "species_richness",
        "shannon_diversity",
        "simpson_diversity",
        "evenness",
        "dominant_species",
        "total_biomass"
    ],
    output_dir="tutorial_results"
)

# Display results
print("\nCalculation results:")
for result in results:
    print(f"  {result.name}: {result.output_path}")
```

## Step 5: Create Maps

Generate publication-ready visualizations:

```python
from gridfia import GridFIA

api = GridFIA()

# Create diversity maps
maps = api.create_maps(
    zarr_path="tutorial_data/wake_forest.zarr",
    map_type="diversity",
    output_dir="tutorial_results/maps",
    dpi=300
)

print(f"Created {len(maps)} map files")
```

## Step 6: Analyze Results

Load and analyze the calculated metrics:

```python
import rasterio
import numpy as np
from pathlib import Path

results_dir = Path("tutorial_results")

# Load species richness
with rasterio.open(results_dir / "species_richness.tif") as src:
    richness = src.read(1)
    valid = richness[richness > 0]

print("Species Richness Statistics:")
print(f"  Mean: {valid.mean():.2f} species")
print(f"  Max: {valid.max()} species")
print(f"  Min: {valid.min()} species")

# Load Shannon diversity
with rasterio.open(results_dir / "shannon_diversity.tif") as src:
    shannon = src.read(1)
    valid = shannon[shannon > 0]

print("\nShannon Diversity Statistics:")
print(f"  Mean: {valid.mean():.3f}")
print(f"  Max: {valid.max():.3f}")
print(f"  Min: {valid.min():.3f}")
```

## Step 7: Identify Diversity Hotspots

Find areas of exceptional biodiversity:

```python
import rasterio
import numpy as np
from scipy import ndimage

# Load Shannon diversity
with rasterio.open("tutorial_results/shannon_diversity.tif") as src:
    shannon = src.read(1)
    transform = src.transform

# Define hotspots as top 10% diversity areas
valid_shannon = shannon[shannon > 0]
threshold = np.percentile(valid_shannon, 90)
hotspots = shannon > threshold

# Clean up with morphological operations
hotspots = ndimage.binary_opening(hotspots, iterations=2)
hotspots = ndimage.binary_closing(hotspots, iterations=2)

# Label connected components
labeled, num_features = ndimage.label(hotspots)
print(f"Found {num_features} diversity hotspots")

# Calculate hotspot areas (30m pixels)
pixel_area_ha = 30 * 30 / 10000  # hectares per pixel
for i in range(1, min(num_features + 1, 6)):  # Top 5
    size = np.sum(labeled == i) * pixel_area_ha
    print(f"  Hotspot {i}: {size:.1f} hectares")
```

## Complete Workflow Example

Here's the entire analysis in one script:

```python
"""
Complete species diversity analysis workflow using GridFIA.
"""
from gridfia import GridFIA
from pathlib import Path

def main():
    # Initialize API
    api = GridFIA()

    # Configuration
    state = "North Carolina"
    county = "Wake"
    species_codes = ["0131", "0068", "0110", "0316", "0611", "0802", "0833"]
    output_dir = Path("diversity_analysis")
    output_dir.mkdir(exist_ok=True)

    # Step 1: Download BIGMAP data
    print("Downloading species data from BIGMAP...")
    files = api.download_species(
        state=state,
        county=county,
        species_codes=species_codes,
        output_dir=output_dir / "downloads"
    )
    print(f"  Downloaded {len(files)} files")

    # Step 2: Create Zarr store
    print("\nCreating Zarr store...")
    zarr_path = api.create_zarr(
        input_dir=output_dir / "downloads",
        output_path=output_dir / "forest.zarr"
    )

    # Validate
    info = api.validate_zarr(zarr_path)
    print(f"  Species: {info['num_species']}")
    print(f"  Shape: {info['shape']}")

    # Step 3: Calculate metrics
    print("\nCalculating diversity metrics...")
    results = api.calculate_metrics(
        zarr_path=zarr_path,
        calculations=[
            "species_richness",
            "shannon_diversity",
            "simpson_diversity",
            "evenness",
            "total_biomass"
        ],
        output_dir=output_dir / "metrics"
    )

    for r in results:
        print(f"  {r.name}: completed")

    # Step 4: Create maps
    print("\nGenerating maps...")
    maps = api.create_maps(
        zarr_path=zarr_path,
        map_type="diversity",
        output_dir=output_dir / "maps"
    )
    print(f"  Created {len(maps)} maps")

    print(f"\nAnalysis complete! Results in: {output_dir}")

if __name__ == "__main__":
    main()
```

## Interpreting Results

### Species Richness (S)

| Value | Interpretation |
|-------|----------------|
| 1-3 | Monoculture or degraded forest |
| 4-7 | Typical managed forest |
| 8+ | Mature, mixed forest ecosystem |

### Shannon Diversity (H')

| Value | Interpretation |
|-------|----------------|
| < 1.0 | Very low diversity, 1-2 species dominate |
| 1.0-2.0 | Low to moderate diversity |
| 2.0-3.0 | Moderate to high diversity, healthy forest |
| > 3.0 | Very high diversity, exceptional biodiversity |

### Simpson Index (1-D)

| Value | Interpretation |
|-------|----------------|
| < 0.5 | Low diversity, few species dominate |
| 0.5-0.7 | Moderate diversity |
| > 0.7 | High diversity |

### Evenness (J)

| Value | Interpretation |
|-------|----------------|
| < 0.5 | Strong dominance by few species |
| 0.5-0.7 | Moderate evenness |
| > 0.7 | High evenness, balanced community |

## Ecological Implications

**High diversity areas** often indicate:

- Mature forest stands
- Ecotone transitions between forest types
- Areas with varied topography or hydrology
- Minimal human disturbance

**Low diversity areas** may indicate:

- Recent disturbance (fire, harvest, disease)
- Plantations or managed stands
- Environmental stress (drought, poor soils)
- Early successional stages

## Example Scripts

Complete working examples are in the `examples/` directory:

| File | Description |
|------|-------------|
| `01_quickstart.py` | Minimal working example |
| `04_calculations.py` | Custom calculation examples |
| `05_species_analysis.py` | Comprehensive species analysis |
| `06_wake_county_full.py` | Full workflow with publication outputs |
| `07_diversity_analysis.py` | Diversity-focused analysis |

## Next Steps

- Try different biomass thresholds for species presence
- Add more species to the analysis
- Compare diversity patterns across counties
- Export results to GIS software (QGIS, ArcGIS)
- Analyze correlation with environmental variables

## References

- Shannon, C.E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379-423.
- Simpson, E.H. (1949). Measurement of diversity. *Nature*, 163(4148), 688.
- Pielou, E.C. (1966). The measurement of diversity in different types of biological collections. *Journal of Theoretical Biology*, 13, 131-144.
- Magurran, A.E. (2004). *Measuring biological diversity*. Blackwell Publishing.
- Wilson, B.T., Knight, J.F., and McRoberts, R.E. (2018). Harmonic regression of Landsat time series for modeling attributes from national forest inventory data. *ISPRS Journal of Photogrammetry and Remote Sensing*, 137: 29-46.

For complete citations and how to cite GridFIA in your work, see [CITATIONS.md](https://github.com/mihiarc/gridfia/blob/main/CITATIONS.md) in the repository.
