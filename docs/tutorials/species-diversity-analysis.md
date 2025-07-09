# Tutorial: Species Diversity Analysis

This tutorial demonstrates how to perform a comprehensive species diversity analysis using BigMap.

## Overview

We'll analyze forest species diversity across North Carolina by:
1. Downloading species biomass data
2. Creating a zarr array for efficient processing
3. Calculating diversity metrics
4. Visualizing the results

## Prerequisites

- BigMap installed (`pip install bigmap` or `uv pip install bigmap`)
- Basic Python knowledge
- ~5GB disk space for data

## Step 1: Download Species Data

First, let's see what species are available:

```bash
bigmap list-species
```

For this tutorial, we'll download common NC tree species:

```bash
# Create data directory
mkdir -p tutorial_data

# Download species data
bigmap download \
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

Convert the downloaded GeoTIFF files to a zarr array:

```python
# create_zarr.py
import zarr
import rasterio
import numpy as np
from pathlib import Path

def create_biomass_zarr(raster_dir, output_path):
    """Create zarr array from species raster files."""
    raster_files = sorted(Path(raster_dir).glob("*.tif"))
    print(f"Found {len(raster_files)} species rasters")
    
    # Read first raster for dimensions and metadata
    with rasterio.open(raster_files[0]) as src:
        height, width = src.shape
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
    
    # Create zarr array with total + species layers
    n_layers = len(raster_files) + 1  # +1 for total biomass
    z = zarr.open_array(
        output_path,
        mode='w',
        shape=(n_layers, height, width),
        chunks=(1, 1000, 1000),
        dtype='f4'
    )
    
    # Load species data and calculate total
    total_biomass = np.zeros((height, width), dtype=np.float32)
    species_codes = ['TOTAL']
    species_names = ['All Species Combined']
    
    for i, raster_file in enumerate(raster_files, 1):
        print(f"Loading {raster_file.name}...")
        with rasterio.open(raster_file) as src:
            data = src.read(1).astype(np.float32)
            z[i] = data
            total_biomass += data
            
            # Extract species info from filename
            species_codes.append(raster_file.stem)
            species_names.append(raster_file.stem)  # Could map to real names
    
    # Store total biomass
    z[0] = total_biomass
    
    # Add metadata
    z.attrs.update({
        'species_codes': species_codes,
        'species_names': species_names,
        'crs': str(crs),
        'transform': list(transform),
        'bounds': list(bounds),
        'units': 'Mg/ha',
        'description': 'NC forest biomass by species'
    })
    
    print(f"Created zarr array: {output_path}")
    print(f"Shape: {z.shape}")
    print(f"Species: {species_codes}")
    
    return output_path

# Create the zarr array
zarr_path = create_biomass_zarr(
    "tutorial_data/", 
    "tutorial_data/nc_biomass.zarr"
)
```

Run the script:
```bash
uv run python create_zarr.py
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
bigmap calculate tutorial_data/nc_biomass.zarr --config diversity_config.yaml
```

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

## Summary

In this tutorial, we:
1. Downloaded species biomass data from the FIA BIGMAP REST API
2. Created an efficient zarr array for processing
3. Calculated multiple diversity metrics (richness, Shannon, Simpson, evenness)
4. Visualized the results as maps
5. Identified diversity hotspots

## Next Steps

- Try different biomass thresholds for species presence
- Add more species to the analysis
- Compare diversity patterns with environmental variables
- Export results for use in GIS software
- Analyze temporal changes if multiple years are available

## Tips

1. **Memory Management**: The chunked processing handles large datasets efficiently
2. **Custom Calculations**: Add your own metrics to the calculation registry
3. **Output Formats**: Use NetCDF for xarray integration, Zarr for large outputs
4. **Visualization**: Export to GeoTIFF for use in QGIS or ArcGIS