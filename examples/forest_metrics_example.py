#!/usr/bin/env python3
"""
Example: Running Forest Metrics Calculations

This example demonstrates how to use the ForestMetricsProcessor
to calculate various forest metrics from zarr biomass data.
"""

from pathlib import Path
import numpy as np
import zarr

from bigmap.config import BigMapSettings, CalculationConfig
from bigmap.core.processors.forest_metrics import ForestMetricsProcessor


def create_example_zarr(output_path: Path) -> str:
    """Create a small example zarr array for demonstration."""
    print("Creating example zarr array...")
    
    # Create zarr array with 3 species + total = 4 layers
    z = zarr.open_array(
        str(output_path),
        mode='w',
        shape=(4, 100, 100),
        chunks=(1, 100, 100),
        dtype='f4'
    )
    
    # Generate example data
    np.random.seed(42)
    
    # Species 1: Oak - dominant
    oak = np.random.rand(100, 100) * 50
    oak[oak < 15] = 0
    
    # Species 2: Pine - common
    pine = np.random.rand(100, 100) * 30
    pine[pine < 20] = 0
    
    # Species 3: Maple - rare
    maple = np.zeros((100, 100))
    maple[30:50, 30:50] = np.random.rand(20, 20) * 20
    
    # Total biomass
    total = oak + pine + maple
    
    # Store in array
    z[0] = total
    z[1] = oak
    z[2] = pine
    z[3] = maple
    
    # Add metadata
    z.attrs.update({
        'species_codes': ['TOTAL', 'OAK', 'PINE', 'MAPLE'],
        'species_names': ['All Species Combined', 'Oak Species', 'Pine Species', 'Maple Species'],
        'crs': 'ESRI:102039',
        'transform': [-2000000, 30, 0, -900000, 0, -30],
        'bounds': [-2000000, -1000000, -1997000, -897000],
        'units': 'Mg/ha',
        'description': 'Example forest biomass data'
    })
    
    print(f"Created example zarr at: {output_path}")
    return str(output_path)


def main():
    """Run forest metrics calculation example."""
    # Setup paths
    example_dir = Path("example_output")
    example_dir.mkdir(exist_ok=True)
    
    zarr_path = example_dir / "example_biomass.zarr"
    
    # Create example data
    zarr_path_str = create_example_zarr(zarr_path)
    
    # Configure calculations
    print("\nConfiguring forest metrics calculations...")
    settings = BigMapSettings(
        output_dir=example_dir / "results",
        calculations=[
            # Species richness - count of species with biomass > 0
            CalculationConfig(
                name="species_richness",
                enabled=True,
                parameters={"biomass_threshold": 0.0},
                output_format="geotiff",
                output_name="species_richness_map"
            ),
            
            # Total biomass across all species
            CalculationConfig(
                name="total_biomass",
                enabled=True,
                output_format="geotiff",
                output_name="total_biomass_map"
            ),
            
            # Shannon diversity index
            CalculationConfig(
                name="shannon_diversity",
                enabled=True,
                output_format="netcdf",
                output_name="shannon_diversity_index"
            ),
            
            # Dominant species identification
            CalculationConfig(
                name="dominant_species",
                enabled=True,
                output_format="geotiff",
                output_name="dominant_species_map"
            ),
            
            # Species proportions (disabled for this example)
            CalculationConfig(
                name="species_proportion",
                enabled=False,
                parameters={"species_indices": [1, 2, 3]}
            )
        ]
    )
    
    # Run calculations
    print("\nRunning forest metrics processor...")
    processor = ForestMetricsProcessor(settings)
    
    # Optional: adjust chunk size for smaller example
    processor.chunk_size = (1, 50, 50)
    
    # Process the data
    results = processor.run_calculations(zarr_path_str)
    
    # Display results
    print("\n✅ Forest metrics calculation completed!")
    print("\nResults saved to:")
    for calc_name, output_path in results.items():
        print(f"  - {calc_name}: {output_path}")
    
    # Demonstrate reading results
    print("\nReading sample results...")
    
    # Read species richness
    import rasterio
    with rasterio.open(results["species_richness"]) as src:
        richness_data = src.read(1)
        print(f"\nSpecies Richness:")
        print(f"  - Min: {richness_data.min()}")
        print(f"  - Max: {richness_data.max()}")
        print(f"  - Mean: {richness_data.mean():.2f}")
    
    # Read Shannon diversity from NetCDF if it exists
    if "shannon_diversity" in results:
        import xarray as xr
        ds = xr.open_dataset(results["shannon_diversity"])
        shannon_data = ds.shannon_diversity.values
        print(f"\nShannon Diversity:")
        print(f"  - Min: {shannon_data.min():.3f}")
        print(f"  - Max: {shannon_data.max():.3f}")
        print(f"  - Mean: {shannon_data.mean():.3f}")
        ds.close()
    else:
        print("\nShannon Diversity: Not saved (likely due to missing netCDF4 dependency)")
    
    print("\n📊 Example complete! Check the 'example_output/results' directory for output files.")


if __name__ == "__main__":
    main()