#!/usr/bin/env python3
"""
Simple demonstration of county-based analysis with BigMap.
"""

from pathlib import Path
from bigmap.utils.location_config import LocationConfig
from bigmap.console import print_info, print_success, print_error


def demo_county_config():
    """Demonstrate creating and using county configurations."""
    
    print("=" * 60)
    print("BigMap County Configuration Demo")
    print("=" * 60)
    
    # Example 1: Create configuration for a specific county
    print("\n1. Creating configuration for Harris County, Texas (Houston area)")
    print("-" * 40)
    
    try:
        # Create config for Harris County, Texas
        config = LocationConfig.from_county("Harris", "Texas")
        
        # Save the configuration
        output_path = Path("config/harris_county_tx.yaml")
        config.save(output_path)
        
        print_success(f"✅ Created configuration for Harris County, TX")
        print(f"   Saved to: {output_path}")
        
        # Display configuration details
        print("\nConfiguration Details:")
        print(f"  Location: {config.location_name}")
        print(f"  Type: {config.location_type}")
        print(f"  Target CRS: {config.target_crs}")
        
    except Exception as e:
        print_error(f"Could not create county config: {e}")
        print_info("Creating manual configuration instead...")
        
        # Create manual configuration with predefined bounds
        config = LocationConfig(location_type="county")
        config._config['location']['name'] = "Harris County, Texas"
        config._config['location']['county'] = "Harris"
        config._config['location']['state'] = "Texas"
        config._config['crs']['target'] = "EPSG:26914"  # Texas Central State Plane
        
        # Harris County approximate bounds (Web Mercator)
        config._config['bounding_boxes']['web_mercator'] = {
            'xmin': -10646000,
            'ymin': 3460000,
            'xmax': -10560000,
            'ymax': 3540000
        }
        
        output_path = Path("config/harris_county_manual.yaml")
        config.save(output_path)
        print_success(f"✅ Created manual configuration for Harris County, TX")
    
    # Example 2: Create configurations for multiple counties
    print("\n2. Creating configurations for multiple counties")
    print("-" * 40)
    
    counties_to_process = [
        ("Wake", "North Carolina"),      # Raleigh area
        ("Orange", "California"),        # Southern California
        ("King", "Washington"),          # Seattle area
        ("Cook", "Illinois"),            # Chicago area
        ("Fulton", "Georgia")            # Atlanta area
    ]
    
    for county, state in counties_to_process:
        try:
            print(f"  Creating config for {county} County, {state}...")
            
            # Create basic config without downloading boundaries
            config = LocationConfig(location_type="county")
            config._config['location']['name'] = f"{county} County, {state}"
            config._config['location']['county'] = county
            config._config['location']['state'] = state
            
            # Detect state plane CRS
            from bigmap.visualization.boundaries import STATE_ABBR
            state_abbr = STATE_ABBR.get(state.lower())
            if state_abbr:
                config._detect_state_plane_crs(state_abbr)
            
            # Save configuration
            filename = f"{county.lower()}_{state.lower().replace(' ', '_')}_config.yaml"
            output_path = Path(f"config/counties/{filename}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            config.save(output_path)
            
            print_success(f"    ✅ Saved: {filename}")
            
        except Exception as e:
            print_error(f"    ❌ Failed for {county}, {state}: {e}")
    
    # Example 3: Show how to use county config for downloads
    print("\n3. Using County Configuration for Data Download")
    print("-" * 40)
    
    print("To download BIGMAP data for a county, use:")
    print("  bigmap download --state Texas --county Harris")
    print("  bigmap download --location-config config/harris_county_tx.yaml")
    
    print("\nOr in Python:")
    print("""
    from bigmap.api import BigMapRestClient
    from bigmap.utils.location_config import LocationConfig
    
    # Load county configuration
    config = LocationConfig("config/harris_county_tx.yaml")
    
    # Download species data
    client = BigMapRestClient()
    client.batch_export_location_species(
        bbox=config.web_mercator_bbox,
        output_dir=Path("data/harris_county"),
        species_codes=['0131', '0068'],  # Loblolly pine, Red maple
        location_name="harris_county_tx"
    )
    """)
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    demo_county_config()