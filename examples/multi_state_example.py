#!/usr/bin/env python3
"""
Example: Working with Multiple States

This example demonstrates how to use BigMap for analyzing
forest data across different US states.
"""

from pathlib import Path
from bigmap.utils.location_config import LocationConfig
from bigmap.api import BigMapRestClient
from bigmap.console import print_info, print_success, print_error


def create_state_configs():
    """Create configuration files for multiple states."""
    states = ['North Carolina', 'Texas', 'California', 'Oregon', 'Georgia']
    configs = []
    
    for state in states:
        print_info(f"Creating configuration for {state}...")
        config = LocationConfig.from_state(state)
        
        # Save to file
        output_path = Path(f"config/{state.lower().replace(' ', '_')}_config.yaml")
        config.save(output_path)
        configs.append(config)
        
        # Print summary
        print_success(f"Created config for {state}")
        print(f"  - Location: {config.location_name}")
        print(f"  - CRS: {config.target_crs}")
        if config.wgs84_bbox:
            bbox = config.wgs84_bbox
            print(f"  - Bounds: ({bbox[0]:.2f}, {bbox[1]:.2f}) to ({bbox[2]:.2f}, {bbox[3]:.2f})")
    
    return configs


def create_county_configs():
    """Create configuration files for specific counties."""
    counties = [
        ('Harris', 'Texas'),        # Houston area
        ('Los Angeles', 'California'),
        ('Cook', 'Illinois'),       # Chicago area
        ('Maricopa', 'Arizona'),    # Phoenix area
        ('King', 'Washington')      # Seattle area
    ]
    
    configs = []
    for county, state in counties:
        print_info(f"Creating configuration for {county} County, {state}...")
        try:
            config = LocationConfig.from_county(county, state)
            
            # Save to file
            output_path = Path(f"config/{county.lower()}_{state.lower()}_config.yaml")
            config.save(output_path)
            configs.append(config)
            
            print_success(f"Created config for {county} County, {state}")
        except Exception as e:
            print_error(f"Failed to create config for {county}, {state}: {e}")
    
    return configs


def download_species_for_state(state_name: str, species_codes: list = None):
    """Download species data for a specific state."""
    print_info(f"\nDownloading data for {state_name}...")
    
    # Create configuration
    config = LocationConfig.from_state(state_name)
    
    # Initialize REST client
    client = BigMapRestClient()
    
    # Use default species codes if none provided
    if species_codes is None:
        species_codes = ['0131', '0068', '0202']  # Loblolly pine, Red maple, Douglas-fir
    
    # Download data
    output_dir = Path(f"data/{state_name.lower().replace(' ', '_')}")
    
    try:
        exported_files = client.batch_export_location_species(
            bbox=config.web_mercator_bbox,
            output_dir=output_dir,
            species_codes=species_codes,
            location_name=state_name.lower().replace(' ', '_')
        )
        
        print_success(f"Downloaded {len(exported_files)} species for {state_name}")
        return exported_files
    except Exception as e:
        print_error(f"Download failed for {state_name}: {e}")
        return []


def compare_state_forests():
    """Compare forest characteristics across states."""
    from bigmap.core.analysis.statistical_analysis import compare_regions
    
    states_to_compare = ['North Carolina', 'Georgia', 'Virginia']
    
    print_info("\nComparing forest characteristics across states...")
    
    # This would require having the data already downloaded and processed
    # For demonstration, we'll just show the structure
    
    comparison_results = {}
    for state in states_to_compare:
        config = LocationConfig.from_state(state)
        comparison_results[state] = {
            'area_sq_km': abs((config.wgs84_bbox[2] - config.wgs84_bbox[0]) * 
                             (config.wgs84_bbox[3] - config.wgs84_bbox[1]) * 111 * 111),
            'location': config.location_name,
            'crs': config.target_crs
        }
    
    print("\nState Comparison:")
    for state, info in comparison_results.items():
        print(f"\n{state}:")
        print(f"  Area: ~{info['area_sq_km']:,.0f} sq km")
        print(f"  CRS: {info['crs']}")


def main():
    """Run multi-state analysis examples."""
    print("=" * 60)
    print("BigMap Multi-State Analysis Example")
    print("=" * 60)
    
    # Create state configurations
    print("\n1. Creating State Configurations")
    print("-" * 40)
    state_configs = create_state_configs()
    
    # Create county configurations
    print("\n2. Creating County Configurations")
    print("-" * 40)
    county_configs = create_county_configs()
    
    # Example: Download data for a state
    print("\n3. Downloading Species Data")
    print("-" * 40)
    # Uncomment to actually download (requires internet connection)
    # download_species_for_state("North Carolina", species_codes=['0131', '0068'])
    print("(Download skipped in example - uncomment to run)")
    
    # Compare states
    print("\n4. Comparing States")
    print("-" * 40)
    compare_state_forests()
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print(f"Created {len(state_configs)} state configs and {len(county_configs)} county configs")
    print("=" * 60)


if __name__ == "__main__":
    main()