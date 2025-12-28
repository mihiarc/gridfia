#!/usr/bin/env python3
"""
Common location bounding boxes for GridFIA examples.

These are pre-calculated bounding boxes for commonly used locations
in the examples. Using explicit bounding boxes ensures examples run
reliably without depending on external boundary services.

All coordinates are in WGS84 (EPSG:4326) format: (west, south, east, north)
"""

# Common counties for examples
COUNTIES = {
    "durham_nc": {
        "name": "Durham County, North Carolina",
        "bbox": (-8796055, 4281816, -8760768, 4333602),  # Full county in Web Mercator
        "crs": "3857",  # Web Mercator EPSG code
        "description": "Full Durham County - compact size ideal for demos (~35x52 km)",
        "bbox_wgs84": (-79.0163, 35.8632, -78.6993, 36.2393)  # For reference
    },
    "wake_nc": {
        "name": "Wake County, North Carolina (subset)",
        "bbox": (-8765000, 4280000, -8740000, 4305000),  # Smaller area in Web Mercator
        "crs": "3857",  # Web Mercator EPSG code
        "description": "Central Wake County - smaller area for quick demos",
        "bbox_wgs84": (-78.72, 35.72, -78.50, 35.90)  # For reference
    },
    "harris_tx": {
        "name": "Harris County, Texas",
        "bbox": (-10688000, 3450000, -10575000, 3537000),  # Web Mercator approximation
        "crs": "3857",
        "description": "Houston metropolitan area",
        "bbox_wgs84": (-95.91, 29.52, -95.01, 30.11)  # For reference
    },
    "king_wa": {
        "name": "King County, Washington",
        "bbox": (-122.54, 47.08, -121.06, 47.78),
        "crs": "4326",
        "description": "Seattle metropolitan area"
    }
}

# Small areas for quick testing
TEST_AREAS = {
    "raleigh_downtown": {
        "name": "Downtown Raleigh",
        "bbox": (-8755000, 4295000, -8750000, 4300000),  # Very small area in Web Mercator
        "crs": "3857",
        "description": "Small area for quick testing",
        "bbox_wgs84": (-78.66, 35.77, -78.61, 35.79)  # For reference
    },
    "mt_hood": {
        "name": "Mt. Hood National Forest",
        "bbox": (-122.0, 45.2, -121.4, 45.6),
        "crs": "4326",
        "description": "Oregon forest area"
    }
}

# State-level bounding boxes (simplified rectangles)
STATES = {
    "north_carolina": {
        "name": "North Carolina",
        "bbox": (-84.32, 33.84, -75.46, 36.59),
        "crs": "4326"
    },
    "texas": {
        "name": "Texas",
        "bbox": (-106.65, 25.84, -93.51, 36.50),
        "crs": "4326"
    },
    "oregon": {
        "name": "Oregon",
        "bbox": (-124.57, 41.99, -116.46, 46.29),
        "crs": "4326"
    }
}


def get_location_bbox(location_key: str) -> tuple:
    """
    Get bounding box for a predefined location.

    Args:
        location_key: Key for the location (e.g., 'wake_nc', 'harris_tx')

    Returns:
        Tuple of (bbox, crs) where bbox is (west, south, east, north)

    Example:
        >>> bbox, crs = get_location_bbox('wake_nc')
        >>> files = api.download_species(bbox=bbox, crs=crs, species_codes=['0131'])
    """
    # Check all location dictionaries
    for locations in [COUNTIES, TEST_AREAS, STATES]:
        if location_key in locations:
            loc = locations[location_key]
            return loc["bbox"], loc["crs"]

    raise ValueError(f"Unknown location: {location_key}. "
                    f"Available: {', '.join(list(COUNTIES.keys()) + list(TEST_AREAS.keys()) + list(STATES.keys()))}")


def list_available_locations():
    """Print all available predefined locations."""
    print("\n=== Available Predefined Locations ===\n")

    print("COUNTIES:")
    for key, info in COUNTIES.items():
        print(f"  {key:12} - {info['name']:30} {info.get('description', '')}")

    print("\nTEST AREAS:")
    for key, info in TEST_AREAS.items():
        print(f"  {key:12} - {info['name']:30} {info.get('description', '')}")

    print("\nSTATES:")
    for key, info in STATES.items():
        print(f"  {key:12} - {info['name']:30}")

    print("\nUsage example:")
    print("  from examples.common_locations import get_location_bbox")
    print("  bbox, crs = get_location_bbox('wake_nc')")
    print("  files = api.download_species(bbox=bbox, crs=crs, species_codes=['0131'])")


if __name__ == "__main__":
    list_available_locations()