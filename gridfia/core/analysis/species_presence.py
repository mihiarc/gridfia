#!/usr/bin/env python3
"""
Analyze which species actually have biomass data in North Carolina.

This script examines each species layer in the zarr array and reports:
- Which species have non-zero biomass pixels
- Coverage statistics for each species
- Summary of species presence in NC

.. note::
   This module is a standalone analysis tool that can be run directly
   or imported for custom species presence analysis. It's primarily
   used for data exploration and validation.

.. todo::
   Integration improvements:
   
   - [ ] Convert to proper CLI command in main GridFIA CLI
   - [ ] Add configuration file support instead of hardcoded paths
   - [ ] Integrate with REST API for dynamic species list
   - [ ] Add export options (CSV, JSON, GeoPackage)
   - [ ] Create unit tests for analysis functions
   - [ ] Add spatial filtering options (by county, bbox)
   - [ ] Support multiple zarr files for comparison
   
   Target Version: v0.3.0
   Priority: Low
   Dependencies: None (standalone utility)

Example Usage::

    # Direct execution
    python -m gridfia.core.analysis.species_presence
    
    # Programmatic usage
    from gridfia.core.analysis import analyze_species_presence
    
    results = analyze_species_presence(
        zarr_path="data/nc_biomass.zarr",
        output_dir="analysis_output"
    )
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

from ...utils.zarr_utils import ZarrStore

def analyze_species_presence(
    zarr_path: str = "output/nc_biomass_expandable.zarr",
    output_dir: str = "output",
    biomass_threshold: float = 0.0
):
    """
    Analyze species presence in North Carolina zarr data.
    
    Parameters:
    -----------
    zarr_path : str
        Path to the zarr array file
    output_dir : str
        Directory to save analysis results
    biomass_threshold : float
        Minimum biomass value to consider as present (default: 0.0)
    """
    
    print("=== Analyzing Species Presence in North Carolina ===\n")

    # Load zarr store using ZarrStore wrapper (Zarr v3 API)
    if not Path(zarr_path).exists():
        print(f"Zarr file not found: {zarr_path}")
        return

    # Open Zarr store using the ZarrStore wrapper class
    zarr_store = ZarrStore.from_path(zarr_path, mode='r')
    species_codes = zarr_store.species_codes
    species_names = zarr_store.species_names

    print(f"Total species in zarr: {len(species_codes)}")
    print(f"Zarr shape: {zarr_store.shape}")
    print(f"Total pixels per species: {zarr_store.shape[1] * zarr_store.shape[2]:,}")
    print()
    
    # Analyze each species
    species_with_data = []
    species_without_data = []
    
    print("Analyzing each species layer...")
    print("="*80)
    
    for i, (code, name) in enumerate(zip(species_codes, species_names)):
        print(f"Processing {i+1}/{len(species_codes)}: {code}", end=" ... ")

        # Load species data from the biomass array using Zarr v3 group access pattern
        data = zarr_store.biomass[i, :, :]
        
        # Calculate statistics
        nonzero_pixels = np.count_nonzero(data > biomass_threshold)
        total_pixels = data.size
        coverage_pct = (nonzero_pixels / total_pixels) * 100
        
        if nonzero_pixels > 0:
            # Calculate additional stats for species with data
            nonzero_data = data[data > 0]
            mean_biomass = nonzero_data.mean()
            max_biomass = nonzero_data.max()
            
            species_with_data.append({
                'index': i,
                'code': code,
                'name': name,
                'pixels': nonzero_pixels,
                'coverage_pct': coverage_pct,
                'mean_biomass': mean_biomass,
                'max_biomass': max_biomass
            })
            print(f"[OK] {nonzero_pixels:,} pixels ({coverage_pct:.3f}%)")
        else:
            species_without_data.append({
                'index': i,
                'code': code,
                'name': name
            })
            print("[NO DATA]")
    
    print("="*80)
    print()
    
    # Summary statistics
    print("SUMMARY STATISTICS")
    print("="*50)
    print(f"Species with biomass data: {len(species_with_data):2d} ({len(species_with_data)/len(species_codes)*100:.1f}%)")
    print(f"Species without data:      {len(species_without_data):2d} ({len(species_without_data)/len(species_codes)*100:.1f}%)")
    print()
    
    # Species WITH data (sorted by coverage)
    if species_with_data:
        print("SPECIES WITH BIOMASS DATA IN NORTH CAROLINA")
        print("="*80)
        species_with_data.sort(key=lambda x: x['coverage_pct'], reverse=True)
        
        print(f"{'#':>2} {'Code':>8} {'Coverage':>10} {'Pixels':>12} {'Mean':>8} {'Max':>8} Species Name")
        print("-" * 80)
        
        for i, species in enumerate(species_with_data, 1):
            print(f"{i:2d} {species['code']:>8} {species['coverage_pct']:>9.3f}% "
                  f"{species['pixels']:>11,} {species['mean_biomass']:>7.1f} "
                  f"{species['max_biomass']:>7.1f} {species['name']}")
    
    print()
    
    # Species WITHOUT data
    if species_without_data:
        print("SPECIES WITHOUT BIOMASS DATA IN NORTH CAROLINA")
        print("="*80)
        print("These species likely don't naturally occur in North Carolina:")
        print()
        
        for i, species in enumerate(species_without_data, 1):
            print(f"{i:2d}. {species['code']}: {species['name']}")
    
    print()
    
    # Save results to output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save species with data to CSV
    if species_with_data:
        import csv
        csv_path = output_path / "species_presence_analysis.csv"
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['rank', 'species_code', 'species_name', 'coverage_pct', 
                         'pixels_with_biomass', 'mean_biomass', 'max_biomass']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for i, species in enumerate(species_with_data, 1):
                writer.writerow({
                    'rank': i,
                    'species_code': species['code'],
                    'species_name': species['name'],
                    'coverage_pct': species['coverage_pct'],
                    'pixels_with_biomass': species['pixels'],
                    'mean_biomass': species['mean_biomass'],
                    'max_biomass': species['max_biomass']
                })
        
        print(f"Results saved to: {csv_path}")
    
    # Top species by coverage
    if len(species_with_data) > 0:
        print("TOP 10 SPECIES BY COVERAGE")
        print("="*50)
        for i, species in enumerate(species_with_data[:10], 1):
            print(f"{i:2d}. {species['name']} ({species['code']}) - {species['coverage_pct']:.3f}%")
    
    # Summary for next steps
    print()
    print("NEXT STEPS")
    print("="*30)
    if len(species_with_data) >= 2:
        print(f"* Use species indices 0-{len(species_with_data)-1} for analysis of NC forest species")
        print(f"* Total forest coverage: {species_with_data[0]['coverage_pct']:.1f}% of NC land area")
        print(f"* Most common species: {species_with_data[1]['name']} ({species_with_data[1]['coverage_pct']:.3f}%)")
    elif len(species_with_data) == 1:
        print(f"* Only 1 species with data found")
        print(f"* Coverage: {species_with_data[0]['coverage_pct']:.1f}% of land area")
    else:
        print("* No species with biomass data found")
    print(f"* Zarr file size: {get_folder_size(zarr_path):.1f} MB")

    # Close the ZarrStore to release resources
    zarr_store.close()

def get_folder_size(folder_path):
    """Calculate total size of folder in MB."""
    import os
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)

if __name__ == "__main__":
    analyze_species_presence() 