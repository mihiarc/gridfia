#!/usr/bin/env python3
"""
Example script demonstrating the flexible calculation framework.

This script shows how to:
1. Configure multiple forest calculations
2. Run calculations programmatically
3. Customize calculation parameters
4. Save results in different formats
"""

import sys
from pathlib import Path
import logging

# Add bigmap to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bigmap.core.forest_metrics import ForestMetricsProcessor, run_forest_analysis
from bigmap.core.calculations import registry
from bigmap.config import BigMapSettings, CalculationConfig

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def example_total_biomass_analysis():
    """Example: Calculate total biomass with custom settings."""
    print("=== Total Biomass Analysis Example ===\n")
    
    # Create custom settings
    settings = BigMapSettings(
        output_dir=Path("output/total_biomass_example"),
        calculations=[
            CalculationConfig(
                name="total_biomass",
                enabled=True,
                output_format="geotiff",
                output_name="nc_total_agb"
            ),
            CalculationConfig(
                name="species_richness",
                enabled=True,
                parameters={"biomass_threshold": 2.0},  # Higher threshold
                output_format="geotiff",
                output_name="nc_richness_mature"
            )
        ]
    )
    
    # Run analysis
    processor = ForestMetricsProcessor(settings)
    zarr_path = "output/nc_biomass_expandable.zarr"
    
    try:
        results = processor.run_analysis_pipeline(zarr_path)
        print(f"Analysis complete! Generated {len(results)} outputs:")
        for name, path in results.items():
            print(f"  {name}: {path}")
    except FileNotFoundError:
        print(f"Zarr file not found: {zarr_path}")
        print("Please run the biomass zarr creation script first.")


def example_diversity_analysis():
    """Example: Comprehensive diversity analysis."""
    print("\n=== Diversity Analysis Example ===\n")
    
    # Load configuration from file
    config_path = "cfg/analysis/diversity_analysis_config.yaml"
    
    try:
        results = run_forest_analysis(
            zarr_path="output/nc_biomass_expandable.zarr",
            config_path=config_path
        )
        
        print(f"Diversity analysis complete! Generated {len(results)} outputs:")
        for name, path in results.items():
            print(f"  {name}: {path}")
            
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please ensure zarr data and config file exist.")


def example_custom_calculation():
    """Example: Define and run a custom calculation."""
    print("\n=== Custom Calculation Example ===\n")
    
    # Define a custom calculation class
    from bigmap.core.calculations import ForestCalculation
    import numpy as np
    
    class BiomassVariability(ForestCalculation):
        """Calculate coefficient of variation across species."""
        
        def __init__(self, **kwargs):
            super().__init__(
                name="biomass_cv",
                description="Coefficient of variation of biomass across species",
                units="coefficient",
                **kwargs
            )
        
        def calculate(self, biomass_data: np.ndarray, **kwargs) -> np.ndarray:
            """Calculate CV = std/mean for each pixel."""
            # Calculate mean and std across species
            mean_biomass = np.mean(biomass_data, axis=0)
            std_biomass = np.std(biomass_data, axis=0)
            
            # Calculate CV, avoiding division by zero
            cv = np.zeros_like(mean_biomass)
            mask = mean_biomass > 0
            cv[mask] = std_biomass[mask] / mean_biomass[mask]
            
            return cv
        
        def validate_data(self, biomass_data: np.ndarray) -> bool:
            return biomass_data.ndim == 3 and biomass_data.shape[0] > 1
    
    # Register the custom calculation
    custom_calc = BiomassVariability()
    registry.register(custom_calc)
    
    print(f"Registered custom calculation: {custom_calc.name}")
    print(f"Available calculations: {registry.list_calculations()}")
    
    # Use in analysis
    settings = BigMapSettings(
        output_dir=Path("output/custom_example"),
        calculations=[
            CalculationConfig(
                name="biomass_cv",
                enabled=True,
                output_format="geotiff"
            ),
            CalculationConfig(
                name="total_biomass",
                enabled=True,
                output_format="geotiff"
            )
        ]
    )
    
    processor = ForestMetricsProcessor(settings)
    zarr_path = "output/nc_biomass_expandable.zarr"
    
    try:
        results = processor.run_analysis_pipeline(zarr_path)
        print(f"Custom analysis complete! Generated {len(results)} outputs:")
        for name, path in results.items():
            print(f"  {name}: {path}")
    except FileNotFoundError:
        print(f"Zarr file not found: {zarr_path}")


def list_available_calculations():
    """List all available calculations."""
    print("\n=== Available Calculations ===\n")
    
    for calc_name in registry.list_calculations():
        info = registry.get_calculation_info(calc_name)
        if info:
            print(f"• {calc_name}")
            print(f"  Description: {info['description']}")
            print(f"  Units: {info['units']}")
            print(f"  Output type: {info['dtype']}")
            print()


def main():
    """Run all examples."""
    print("BigMap Flexible Calculation Framework Examples")
    print("=" * 50)
    
    # List available calculations
    list_available_calculations()
    
    # Run examples
    example_total_biomass_analysis()
    example_diversity_analysis()
    example_custom_calculation()
    
    print("\n" + "=" * 50)
    print("Examples complete! Check the output directories for results.")
    print("\nTo create your own analysis:")
    print("1. Define calculations in a YAML config file")
    print("2. Use run_forest_analysis(zarr_path, config_path)")
    print("3. Or create custom ForestCalculation classes")


if __name__ == "__main__":
    main()