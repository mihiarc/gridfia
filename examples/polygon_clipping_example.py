"""
Example: Using Custom Polygon Boundaries for Data Download and Clipping

This example demonstrates how to:
1. Use a custom polygon boundary for data downloads
2. Automatically clip downloaded data to the polygon shape
3. Use county boundaries with actual shape clipping (not just bbox)
"""

from pathlib import Path
from bigmap import BigMapAPI
import geopandas as gpd

# Initialize API
api = BigMapAPI()

# =============================================================================
# Example 1: Using a Custom Polygon File
# =============================================================================
print("\n" + "="*70)
print("Example 1: Download and clip using custom polygon")
print("="*70)

# You can use GeoJSON, Shapefile, or any format supported by GeoPandas
polygon_file = "study_area.geojson"  # Your polygon file

# Download species data - downloads bbox and clips to actual polygon
files = api.download_species(
    polygon=polygon_file,
    species_codes=["0202", "0122"],  # Douglas-fir, Ponderosa Pine
    output_dir="downloads/polygon_study"
)

# Create Zarr with automatic clipping
zarr_path = api.create_zarr(
    input_dir="downloads/polygon_study",
    output_path="data/polygon_study.zarr",
    clip_to_polygon=True  # Auto-detects polygon from saved config
)

print(f"Created clipped Zarr store: {zarr_path}")

# =============================================================================
# Example 2: County Boundaries with Actual Shape Clipping
# =============================================================================
print("\n" + "="*70)
print("Example 2: Download county data with boundary clipping")
print("="*70)

# Download for Lane County, Oregon with actual boundary clipping
files = api.download_species(
    state="Oregon",
    county="Lane",
    species_codes=["0202", "0122"],
    use_boundary_clip=True,  # Store and use actual county boundary
    output_dir="downloads/lane_county"
)

# Create Zarr - will automatically clip to county boundary
zarr_path = api.create_zarr(
    input_dir="downloads/lane_county",
    output_path="data/lane_county_clipped.zarr",
    clip_to_polygon=True
)

print(f"Created county-clipped Zarr store: {zarr_path}")

# Calculate metrics on the clipped data
results = api.calculate_metrics(
    zarr_path,
    calculations=["species_richness", "shannon_diversity", "total_biomass"]
)

for result in results:
    print(f"\nCalculated {result.name}: {result.output_path}")

# =============================================================================
# Example 3: Using a GeoDataFrame Directly
# =============================================================================
print("\n" + "="*70)
print("Example 3: Using GeoDataFrame for custom area")
print("="*70)

# Load and subset a larger dataset to get a specific polygon
# For example, select parcels from a geopackage
parcel_file = "merged_tim_FACTnobids_neversold_only.gpkg"

if Path(parcel_file).exists():
    # Load a few parcels as our study area
    gdf = gpd.read_file(parcel_file)

    # Select first 10 parcels as study area (just as an example)
    study_area = gdf.head(10)

    # Download and clip using this GeoDataFrame
    files = api.download_species(
        polygon=study_area,
        species_codes=["0202"],
        output_dir="downloads/parcels"
    )

    # Create clipped Zarr
    zarr_path = api.create_zarr(
        input_dir="downloads/parcels",
        output_path="data/parcels.zarr",
        clip_to_polygon=study_area  # Pass GeoDataFrame directly
    )

    print(f"Created parcel-clipped Zarr store: {zarr_path}")

# =============================================================================
# Example 4: Creating a Location Configuration with Polygon
# =============================================================================
print("\n" + "="*70)
print("Example 4: Creating and reusing location configurations")
print("="*70)

# Create a location config from polygon for reuse
config = api.get_location_config(
    polygon="study_area.geojson",
    output_path="configs/my_study_area.yaml"
)

print(f"Configuration saved to: configs/my_study_area.yaml")
print(f"Location: {config.location_name}")
print(f"Has polygon boundary: {config.has_polygon}")
print(f"Bounding box: {config.wgs84_bbox}")

# Later, reuse this config
files = api.download_species(
    location_config="configs/my_study_area.yaml",
    species_codes=["0122"]
)

# =============================================================================
# Example 5: Manual Polygon Clipping
# =============================================================================
print("\n" + "="*70)
print("Example 5: Manual polygon clipping of existing GeoTIFFs")
print("="*70)

from bigmap.utils.polygon_utils import clip_geotiffs_batch

# If you already have downloaded GeoTIFFs and want to clip them
clipped_files = clip_geotiffs_batch(
    input_dir="downloads/existing_species",
    polygon="study_area.geojson",
    output_dir="downloads/clipped_species"
)

print(f"Clipped {len(clipped_files)} files")

# =============================================================================
# Workflow Summary
# =============================================================================
print("\n" + "="*70)
print("WORKFLOW SUMMARY")
print("="*70)
print("""
The typical workflow is:

1. Provide a polygon boundary (GeoJSON, Shapefile, or GeoDataFrame)
2. Download species data - system downloads bbox and saves polygon config
3. Create Zarr with clip_to_polygon=True - automatically clips to polygon
4. Analyze the clipped data using standard BigMap methods

Benefits:
- Reduces storage by excluding areas outside your region of interest
- More accurate statistics for irregular study areas
- Cleaner visualizations showing only relevant areas
- Works with any polygon format supported by GeoPandas
""")
