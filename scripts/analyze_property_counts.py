#!/usr/bin/env python3

import geopandas as gpd
import pandas as pd
from pathlib import Path

def analyze_property_counts():
    """Analyze property counts at each filtering step."""
    print("\nAnalyzing property counts...")
    
    # Load heirs properties
    heirs_df = gpd.read_parquet("data/raw/nc-hp_v2.parquet")
    print(f"\nTotal heirs properties in NC: {len(heirs_df)}")
    
    # Analyze county names
    print("\nUnique county names in dataset:")
    county_counts = heirs_df['county_nam'].value_counts()
    for county, count in county_counts.items():
        print(f"  {county}: {count}")
    
    # Check Vance County variations
    vance_variations = ['Vance County', 'VANCE COUNTY', 'Vance', 'VANCE']
    print("\nChecking for Vance County variations:")
    for variation in vance_variations:
        count = len(heirs_df[heirs_df['county_nam'] == variation])
        print(f"  '{variation}': {count} properties")
    
    # Analyze Vance County properties in detail
    vance_mask = heirs_df['county_nam'].str.contains('VANCE|Vance', case=False, na=False)
    vance_properties = heirs_df[vance_mask]
    print(f"\nTotal Vance County properties (any variation): {len(vance_properties)}")
    
    if len(vance_properties) > 0:
        print("\nVance County Property Statistics:")
        print(f"  Average land area: {vance_properties['land_acre'].mean():.2f} acres")
        print(f"  Total land area: {vance_properties['land_acre'].sum():.2f} acres")
        print(f"  Properties with HPbin=1: {len(vance_properties[vance_properties['HPbin'] == 1])}")
        print(f"  Properties with heir500=1: {len(vance_properties[vance_properties['heir500'] == 1])}")
    
    # Check for null or empty county names
    null_counties = heirs_df[heirs_df['county_nam'].isna()]
    empty_counties = heirs_df[heirs_df['county_nam'] == '']
    print(f"\nProperties with null county: {len(null_counties)}")
    print(f"Properties with empty county: {len(empty_counties)}")
    
    # Save detailed analysis
    output_file = Path('data/processed/property_count_analysis.json')
    analysis = {
        'total_properties': len(heirs_df),
        'county_counts': county_counts.to_dict(),
        'null_counties': len(null_counties),
        'empty_counties': len(empty_counties),
        'vance_variations': {
            var: len(heirs_df[heirs_df['county_nam'] == var])
            for var in vance_variations
        },
        'vance_total': len(vance_properties),
        'vance_statistics': {
            'mean_acres': float(vance_properties['land_acre'].mean()),
            'total_acres': float(vance_properties['land_acre'].sum()),
            'hpbin_count': int(len(vance_properties[vance_properties['HPbin'] == 1])),
            'heir500_count': int(len(vance_properties[vance_properties['heir500'] == 1]))
        } if len(vance_properties) > 0 else None
    }
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        import json
        json.dump(analysis, f, indent=2)
    
    print(f"\nDetailed analysis saved to: {output_file}")

if __name__ == '__main__':
    analyze_property_counts() 