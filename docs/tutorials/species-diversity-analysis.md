# Tutorial: Species Diversity Analysis

This tutorial demonstrates how to perform a comprehensive species diversity analysis using GridFIA.

## Scientific Background

Species diversity is a fundamental measure of ecosystem health and resilience. This tutorial covers three key diversity metrics:

### Shannon Diversity Index (H')
The Shannon diversity index (Shannon, 1948) measures both species richness and evenness:

**H' = -Σ(pi × ln(pi))**

Where pi is the proportion of species i. Higher values indicate greater diversity.
- Values typically range from 0 to 5
- H' = 0 indicates a monoculture
- H' > 3 indicates high diversity

### Simpson Diversity Index
The Simpson index (Simpson, 1949) has multiple formulations:

**Simpson's Dominance (D)**: Σ(pi²)
- Probability that two individuals belong to the same species
- Values range from 0 to 1 (lower = more diverse)

**Simpson's Diversity (1-D)**: 1 - Σ(pi²)
- Probability that two individuals belong to different species
- Values range from 0 to 1 (higher = more diverse)

**Inverse Simpson (1/D)**: 1/Σ(pi²)
- Effective number of equally abundant species
- Values range from 1 to S (number of species)

Note: The GridFIA implementation calculates dominance (D) by default, with options for diversity (1-D) or inverse (1/D) via the `inverse` parameter.

### Pielou's Evenness (J)
Pielou's evenness (Pielou, 1966) measures how evenly species are distributed:

**J = H' / ln(S)**

Where S is the number of species.
- Values range from 0 to 1
- J = 1 indicates perfect evenness
- J < 0.5 suggests dominance by few species

### When to Use Each Index
- **Shannon**: General biodiversity assessment, sensitive to rare species
- **Simpson**: When dominance patterns are important
- **Species Richness**: Simple count when presence/absence is sufficient
- **Evenness**: To assess community balance independent of richness

## Overview

We'll analyze forest species diversity across North Carolina by:
1. Downloading species biomass data
2. Creating a zarr array for efficient processing
3. Calculating diversity metrics
4. Visualizing and interpreting the results

## Prerequisites

- GridFIA installed (`pip install gridfia` or `uv pip install gridfia`)
- Basic Python knowledge
- ~5GB disk space for data

## Example Code

Complete working examples are available in the `examples/` directory:
- **Quick start**: See `examples/01_quickstart.py` for a minimal example
- **Species analysis**: See `examples/05_species_analysis.py` for comprehensive species analysis
- **Full workflow**: See `examples/06_wake_county_full.py` for complete case study

## Step 1: Download Species Data

First, let's see what species are available:

```bash
gridfia list-species
```

For this tutorial, we'll download common NC tree species:

```bash
# Create data directory
mkdir -p tutorial_data

# Download species data
gridfia download \
    --species 0131 \  # Loblolly pine
    --species 0068 \  # Eastern white pine  
    --species 0110 \  # Shortleaf pine
    --species 0316 \  # Eastern redcedar
    --species 0611 \  # Sweetgum
    --species 0802 \  # White oak
    --species 0833 \  # Northern red oak
    --output tutorial_data/
```

## Step 2: Create Zarr Array

Convert the downloaded GeoTIFF files to a zarr array.

**See `examples/utils.py`** for the reusable `create_zarr_from_rasters()` function.

```python
# Using the shared utility function
from examples.utils import create_zarr_from_rasters
from pathlib import Path

# Create the zarr array
zarr_path = create_zarr_from_rasters(
    raster_dir=Path("tutorial_data/"),
    output_path=Path("tutorial_data/nc_biomass.zarr"),
    chunk_size=(1, 1000, 1000)
)

print(f"Created zarr array: {zarr_path}")
```

Or use the GridFIA API directly:
```python
from gridfia import GridFIA

api = GridFIA()
zarr_path = api.create_zarr(
    input_dir="tutorial_data/",
    output_path="tutorial_data/nc_biomass.zarr"
)
```

## Step 3: Configure Diversity Analysis

Create a configuration file for diversity analysis:

```yaml
# diversity_config.yaml
app_name: NC Forest Diversity Analysis
output_dir: tutorial_results/diversity

calculations:
  # Species count per pixel
  - name: species_richness
    enabled: true
    parameters:
      biomass_threshold: 0.5  # Minimum Mg/ha to count as present
    output_format: geotiff
    
  # Shannon diversity index
  - name: shannon_diversity
    enabled: true
    parameters:
      base: e  # Natural logarithm
    output_format: geotiff
    
  # Simpson diversity index  
  - name: simpson_diversity
    enabled: true
    output_format: geotiff
    
  # Species evenness
  - name: evenness
    enabled: true
    output_format: geotiff
    
  # Dominant species map
  - name: dominant_species
    enabled: true
    output_format: geotiff
    
  # Total biomass for context
  - name: total_biomass
    enabled: true
    output_format: geotiff
```

## Step 4: Run Diversity Calculations

Execute the diversity analysis:

```bash
gridfia calculate tutorial_data/nc_biomass.zarr --config diversity_config.yaml
```

**See `examples/04_calculations.py`** for detailed calculation examples and custom metrics.

## Step 5: Visualize Results

Create a Python script to visualize the diversity maps:

```python
# visualize_diversity.py
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set up plot style
plt.style.use('seaborn-v0_8-darkgrid')

# Load results
results_dir = Path("tutorial_results/diversity")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

# Define visualization settings
plots = [
    ("species_richness.tif", "Species Richness", "viridis", "Number of Species"),
    ("shannon_diversity.tif", "Shannon Diversity Index", "plasma", "H'"),
    ("simpson_diversity.tif", "Simpson Diversity Index", "cividis", "1-D"),
    ("evenness.tif", "Species Evenness", "RdYlBu", "Pielou's J"),
    ("dominant_species.tif", "Dominant Species", "tab20", "Species ID"),
    ("total_biomass.tif", "Total Biomass", "YlGn", "Mg/ha")
]

for ax, (filename, title, cmap, label) in zip(axes, plots):
    filepath = results_dir / filename
    
    with rasterio.open(filepath) as src:
        data = src.read(1)
        
        # Handle no-data values
        data = np.ma.masked_where(data == src.nodata, data)
        
        # Plot
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(label, rotation=270, labelpad=20)

plt.suptitle('North Carolina Forest Diversity Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('nc_forest_diversity.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\nDiversity Statistics Summary:")
print("-" * 50)

with rasterio.open(results_dir / "species_richness.tif") as src:
    richness = src.read(1)
    valid_richness = richness[richness > 0]
    print(f"Species Richness:")
    print(f"  Mean: {valid_richness.mean():.2f} species")
    print(f"  Max: {valid_richness.max()} species")
    print(f"  Min: {valid_richness.min()} species")

with rasterio.open(results_dir / "shannon_diversity.tif") as src:
    shannon = src.read(1)
    valid_shannon = shannon[shannon > 0]
    print(f"\nShannon Diversity:")
    print(f"  Mean: {valid_shannon.mean():.3f}")
    print(f"  Max: {valid_shannon.max():.3f}")
    print(f"  Min: {valid_shannon.min():.3f}")
```

Run the visualization:
```bash
uv run python visualize_diversity.py
```

## Step 6: Advanced Analysis

Let's identify diversity hotspots:

```python
# diversity_hotspots.py
import rasterio
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

# Load diversity indices
with rasterio.open("tutorial_results/diversity/shannon_diversity.tif") as src:
    shannon = src.read(1)
    transform = src.transform

# Define hotspots as areas with high diversity
threshold = np.percentile(shannon[shannon > 0], 90)  # Top 10%
hotspots = shannon > threshold

# Apply morphological operations to clean up
hotspots = ndimage.binary_opening(hotspots, iterations=2)
hotspots = ndimage.binary_closing(hotspots, iterations=2)

# Label connected components
labeled, num_features = ndimage.label(hotspots)
print(f"Found {num_features} diversity hotspots")

# Calculate hotspot statistics
hotspot_sizes = []
for i in range(1, num_features + 1):
    size = np.sum(labeled == i)
    hotspot_sizes.append(size * 30 * 30 / 10000)  # Convert to hectares

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Shannon diversity map
im1 = ax1.imshow(shannon, cmap='viridis')
ax1.set_title('Shannon Diversity Index')
plt.colorbar(im1, ax=ax1)

# Hotspots overlay
ax2.imshow(shannon, cmap='gray', alpha=0.5)
ax2.imshow(np.ma.masked_where(labeled == 0, labeled), cmap='hot')
ax2.set_title(f'Diversity Hotspots (Top 10%, n={num_features})')

plt.tight_layout()
plt.savefig('diversity_hotspots.png', dpi=300)
plt.show()

# Print statistics
print(f"\nHotspot Statistics:")
print(f"Total area: {sum(hotspot_sizes):.1f} hectares")
print(f"Average size: {np.mean(hotspot_sizes):.1f} hectares")
print(f"Largest hotspot: {max(hotspot_sizes):.1f} hectares")
```

## Interpreting Results

### Understanding Diversity Values

**Species Richness (S)**
- **Low (1-3)**: Monoculture or degraded forest
- **Medium (4-7)**: Typical managed forest
- **High (8+)**: Mature, mixed forest ecosystem

**Shannon Diversity (H')**
- **< 1.0**: Very low diversity, dominated by 1-2 species
- **1.0-2.0**: Low to moderate diversity
- **2.0-3.0**: Moderate to high diversity, healthy forest
- **> 3.0**: Very high diversity, exceptional biodiversity

**Simpson Index**
- **Dominance (D < 0.5)**: High diversity
- **Dominance (D > 0.7)**: Low diversity, few species dominate
- **Diversity (1-D > 0.5)**: Good diversity
- **Inverse (1/D > 5)**: High effective species number

**Evenness (J)**
- **< 0.5**: Strong dominance by few species
- **0.5-0.7**: Moderate evenness
- **> 0.7**: High evenness, balanced community

### Ecological Implications

High diversity areas often indicate:
- Mature forest stands
- Ecotone transitions between forest types
- Areas with varied topography or hydrology
- Minimal human disturbance

Low diversity areas may indicate:
- Recent disturbance (fire, harvest, disease)
- Plantations or managed stands
- Environmental stress (drought, poor soils)
- Early successional stages

## Summary

In this tutorial, we:
1. Downloaded species biomass data from the FIA BIGMAP REST API
2. Created an efficient zarr array for processing
3. Calculated multiple diversity metrics (richness, Shannon, Simpson, evenness)
4. Visualized the results as maps
5. Identified diversity hotspots
6. Learned to interpret diversity metrics in ecological context

## Complete Examples

For complete, runnable code:
- **`examples/01_quickstart.py`** - Minimal working example
- **`examples/05_species_analysis.py`** - Comprehensive species and diversity analysis
- **`examples/06_wake_county_full.py`** - Full workflow with publication outputs

## Next Steps

- Try different biomass thresholds for species presence
- Add more species to the analysis
- Compare diversity patterns with environmental variables
- Export results for use in GIS software
- Analyze temporal changes if multiple years are available

## Tips

1. **Memory Management**: The chunked processing handles large datasets efficiently
2. **Custom Calculations**: See `examples/04_calculations.py` for custom metrics
3. **Output Formats**: Use NetCDF for xarray integration, Zarr for large outputs
4. **Visualization**: Export to GeoTIFF for use in QGIS or ArcGIS

## References

- Shannon, C.E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379-423.
- Simpson, E.H. (1949). Measurement of diversity. *Nature*, 163(4148), 688.
- Pielou, E.C. (1966). The measurement of diversity in different types of biological collections. *Journal of Theoretical Biology*, 13, 131-144.
- Magurran, A.E. (2004). *Measuring biological diversity*. Blackwell Publishing.
- USDA Forest Service. (2018). *BIGMAP 2018 Forest Biomass Dataset*. Forest Inventory and Analysis Program.

For complete citations and how to cite GridFIA in your work, see [CITATIONS.md](../../CITATIONS.md).