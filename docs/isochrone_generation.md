# Isochrone Generation with SocialMapper

This document explains how to generate isochrones (travel time areas) for the Montana timber mill site using the SocialMapper package.

## What is SocialMapper?

SocialMapper is a Python toolkit that helps analyze accessibility patterns by generating isochrones - areas reachable within a certain travel time from a point of interest. It supports multiple travel modes (driving, walking, biking) and can integrate with census data for demographic analysis.

## Installation

First, ensure SocialMapper is installed:

```bash
pip install socialmapper
```

Or with uv:

```bash
uv pip install socialmapper
```

## Generating Isochrones for the Mill Site

### Method 1: Using the Full-Featured Script

The `generate_mill_isochrones.py` script provides a complete interface for generating isochrones:

```bash
# Generate default 120-minute driving isochrone
python scripts/generate_mill_isochrones.py

# Generate multiple travel times
python scripts/generate_mill_isochrones.py -t 60 -t 90 -t 120

# Generate walking isochrones
python scripts/generate_mill_isochrones.py -t 15 -t 30 --mode walk

# Generate biking isochrones
python scripts/generate_mill_isochrones.py -t 30 -t 60 --mode bike
```

### Method 2: Using the Simple Script

The `generate_isochrones_simple.py` script demonstrates direct isochrone generation:

```bash
python scripts/generate_isochrones_simple.py
```

This script:
- Generates a single 120-minute driving isochrone
- Creates multiple isochrones for different travel times
- Saves results as GeoJSON files

### Method 3: Using SocialMapper Directly

You can also use SocialMapper's API directly in your own scripts:

```python
from socialmapper.isochrone import create_isochrones_from_poi_list, TravelMode

# Prepare POI data
poi_data = {
    'poi_ids': ['mill_001'],
    'poi_names': ['Montana Mill Site'],
    'latitudes': [47.167012],
    'longitudes': [-113.466881],
    'poi_count': 1
}

# Generate 120-minute driving isochrone
result = create_isochrones_from_poi_list(
    poi_data=poi_data,
    travel_time_limit=120,
    travel_mode=TravelMode.DRIVE,
    combine_results=True,
    save_individual_files=False
)
```

## Understanding the Output

The isochrone files contain:
- **Geometry**: Polygon showing the reachable area
- **Metadata**: Travel time, travel mode, POI information
- **Coordinates**: In WGS84 (EPSG:4326) by default

### Output Formats

- **GeoJSON**: Human-readable, widely supported format
- **GeoParquet**: Efficient binary format for large datasets

## Travel Modes

SocialMapper supports three travel modes:

1. **Drive**: Uses road networks accessible by cars
   - Default speed: 50 km/h (can vary by road type)
   - Considers highways, major roads, local streets

2. **Walk**: Uses pedestrian-accessible paths
   - Default speed: 5 km/h
   - Includes sidewalks, footpaths, crosswalks

3. **Bike**: Uses bike-friendly routes
   - Default speed: 15 km/h
   - Includes bike lanes, shared roads, trails

## Integration with Existing Analysis

The generated isochrones can be used with the existing analysis scripts:

1. `06_analyze_mill_isochrone_biomass.py` - Analyzes forest biomass within the isochrone
2. `deprecated_07_visualize_mill_isochrone.py` - Creates maps with the isochrone overlay

## Troubleshooting

### Common Issues

1. **"SocialMapper not found"**
   - Solution: Install with `pip install socialmapper`

2. **"Network data not available"**
   - The area may lack sufficient road/path data
   - Try a different travel mode or smaller travel time

3. **"Invalid coordinates"**
   - Ensure coordinates are in decimal degrees
   - Check that latitude is between -90 and 90
   - Check that longitude is between -180 and 180

### Performance Tips

- Larger travel times take longer to compute
- Rural areas may have sparse road networks
- Consider using GeoParquet format for better performance

## Advanced Usage

### Custom Speed Settings

SocialMapper uses default speeds, but you can customize them through environment variables or configuration files. See the SocialMapper documentation for details.

### Batch Processing

For multiple locations, create a CSV file with all POIs and use SocialMapper's batch processing capabilities:

```csv
name,latitude,longitude,type
Mill Site 1,47.167012,-113.466881,mill
Mill Site 2,47.234567,-113.345678,mill
```

### Combining with Census Data

SocialMapper can also retrieve census data for the areas within isochrones. This is useful for understanding the population and demographics of the accessible area.

## References

- [SocialMapper Documentation](https://mihiarc.github.io/socialmapper)
- [SocialMapper GitHub Repository](https://github.com/mihiarc/socialmapper)
- [OpenStreetMap](https://www.openstreetmap.org) (source of road network data)