#!/usr/bin/env python
"""
Simple example of generating isochrones using SocialMapper's lower-level API.

This script demonstrates direct isochrone generation without census analysis.
"""

import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Import SocialMapper's isochrone module directly
try:
    from socialmapper.isochrone import create_isochrones_from_poi_list, TravelMode
except ImportError:
    print("Error: SocialMapper package not found.")
    print("Please install it with: pip install socialmapper")
    sys.exit(1)


def generate_isochrone_simple():
    """
    Generate isochrones using SocialMapper's direct isochrone function.
    """
    print("Generating isochrones for Montana timber mill site...")
    
    # Load mill location
    mill_csv = Path("config/montana_timber_mill_location.csv")
    mill_df = pd.read_csv(mill_csv)
    mill_site = mill_df[mill_df['type'] == 'mill'].iloc[0]
    
    print(f"\nMill Location: {mill_site['name']}")
    print(f"Coordinates: ({mill_site['lat']:.6f}, {mill_site['lon']:.6f})")
    
    # Create POI data dictionary in the format expected by SocialMapper
    poi_data = {
        'poi_ids': ['mill_site_001'],
        'poi_names': [mill_site['name']],
        'latitudes': [mill_site['lat']],
        'longitudes': [mill_site['lon']],
        'poi_count': 1
    }
    
    # Generate 120-minute driving isochrone
    travel_time = 120
    travel_mode = TravelMode.DRIVE
    
    print(f"\nGenerating {travel_time}-minute {travel_mode.value} isochrone...")
    
    try:
        # Generate isochrone
        result = create_isochrones_from_poi_list(
            poi_data=poi_data,
            travel_time_limit=travel_time,
            combine_results=True,  # Combine if multiple POIs
            save_individual_files=False,  # Return GeoDataFrame instead of saving
            use_parquet=True,
            travel_mode=travel_mode,
        )
        
        # Handle the result
        if isinstance(result, gpd.GeoDataFrame):
            isochrone_gdf = result
        elif isinstance(result, str):
            # It's a file path
            isochrone_gdf = gpd.read_parquet(result)
        else:
            print(f"Unexpected result type: {type(result)}")
            return
        
        # Save the isochrone
        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as GeoJSON
        output_path = output_dir / f"mill_isochrone_{travel_time}min_simple.geojson"
        isochrone_gdf.to_file(output_path, driver='GeoJSON')
        print(f"\n✓ Saved isochrone to: {output_path}")
        
        # Print some statistics
        if len(isochrone_gdf) > 0:
            geom = isochrone_gdf.iloc[0].geometry
            # Calculate area (assuming the geometry is in a projected CRS)
            area_sqm = geom.area
            area_sqkm = area_sqm / 1e6 if area_sqm > 0 else 0
            print(f"  Isochrone area: ~{area_sqkm:.2f} km²")
            print(f"  Geometry type: {geom.geom_type}")
        
    except Exception as e:
        print(f"\nError generating isochrone: {e}")
        import traceback
        traceback.print_exc()


def generate_multiple_isochrones():
    """
    Generate isochrones for multiple travel times and modes.
    """
    print("\n" + "="*50)
    print("Generating multiple isochrones...")
    
    # Load mill location
    mill_csv = Path("config/montana_timber_mill_location.csv")
    mill_df = pd.read_csv(mill_csv)
    mill_site = mill_df[mill_df['type'] == 'mill'].iloc[0]
    
    # Create POI data
    poi_data = {
        'poi_ids': ['mill_site_001'],
        'poi_names': [mill_site['name']],
        'latitudes': [mill_site['lat']],
        'longitudes': [mill_site['lon']],
        'poi_count': 1
    }
    
    # Define combinations to generate
    scenarios = [
        (60, TravelMode.DRIVE, "1 hour drive"),
        (90, TravelMode.DRIVE, "1.5 hour drive"),
        (120, TravelMode.DRIVE, "2 hour drive"),
    ]
    
    output_dir = Path("output/isochrones_multi")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for travel_time, mode, description in scenarios:
        print(f"\nGenerating {description} isochrone...")
        
        try:
            result = create_isochrones_from_poi_list(
                poi_data=poi_data,
                travel_time_limit=travel_time,
                combine_results=True,
                save_individual_files=False,
                use_parquet=True,
                travel_mode=mode,
            )
            
            if isinstance(result, gpd.GeoDataFrame):
                isochrone_gdf = result
            elif isinstance(result, str):
                isochrone_gdf = gpd.read_parquet(result)
            else:
                continue
            
            # Add metadata
            isochrone_gdf['travel_time'] = travel_time
            isochrone_gdf['travel_mode'] = mode.value
            isochrone_gdf['description'] = description
            
            results.append(isochrone_gdf)
            
            # Save individual file
            filename = f"isochrone_{travel_time}min_{mode.value}.geojson"
            output_path = output_dir / filename
            isochrone_gdf.to_file(output_path, driver='GeoJSON')
            print(f"  ✓ Saved: {filename}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Combine all isochrones into one file
    if results:
        combined_gdf = gpd.GeoDataFrame(pd.concat(results, ignore_index=True))
        combined_path = output_dir / "all_isochrones.geojson"
        combined_gdf.to_file(combined_path, driver='GeoJSON')
        print(f"\n✓ Saved combined file: {combined_path}")


if __name__ == "__main__":
    # Simple single isochrone
    generate_isochrone_simple()
    
    # Multiple isochrones
    generate_multiple_isochrones()