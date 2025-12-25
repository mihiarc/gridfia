#!/usr/bin/env python3
"""
Forest Calculations Example

Demonstrates the flexible calculation framework:
- Built-in calculations
- Custom calculations
- Different output formats
- Configuration patterns
"""

from pathlib import Path
import numpy as np
from gridfia.examples import create_sample_zarr, print_zarr_info, AnalysisConfig
from gridfia.config import GridFIASettings, CalculationConfig
from gridfia.core.processors.forest_metrics import ForestMetricsProcessor
from gridfia.core.calculations import ForestCalculation, registry
from rich.console import Console
from rich.table import Table

console = Console()


def show_available_calculations():
    """List all available calculations."""
    console.print("\n[bold blue]Available Calculations[/bold blue]")
    console.print("-" * 40)

    table = Table(title="Built-in Forest Calculations")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="yellow")
    table.add_column("Units", style="green")

    calculations = [
        ("species_richness", "Count of species with biomass > threshold", "count"),
        ("shannon_diversity", "Shannon diversity index (H')", "index"),
        ("simpson_diversity", "Simpson diversity index (1-D)", "index"),
        ("evenness", "Pielou's evenness (J)", "ratio"),
        ("total_biomass", "Sum of all species biomass", "Mg/ha"),
        ("dominant_species", "ID of species with highest biomass", "species_id"),
        ("species_proportion", "Proportion of specific species", "ratio"),
        ("species_percentage", "Percentage of specific species", "percent"),
    ]

    for name, desc, units in calculations:
        table.add_row(name, desc, units)

    console.print(table)


def example_basic_calculations():
    """Run basic diversity calculations."""
    console.print("\n[bold blue]Basic Calculations[/bold blue]")
    console.print("-" * 40)

    # Create sample data
    zarr_path = create_sample_zarr(Path("temp_calculations.zarr"), n_species=5)

    # Configure basic calculations
    settings = GridFIASettings(
        output_dir=Path("results/basic"),
        calculations=[
            CalculationConfig(
                name="species_richness",
                enabled=True,
                parameters={"biomass_threshold": 0.5},
                output_format="geotiff"
            ),
            CalculationConfig(
                name="shannon_diversity",
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

    # Run calculations
    processor = ForestMetricsProcessor(settings)
    results = processor.run_calculations(str(zarr_path))

    console.print(f"\n✅ Completed {len(results)} calculations:")
    for name, path in results.items():
        console.print(f"   - {name}: {Path(path).name}")

    # Clean up
    import shutil
    shutil.rmtree(zarr_path)
    if Path("results").exists():
        shutil.rmtree("results")


def example_custom_calculation():
    """Create and register a custom calculation."""
    console.print("\n[bold blue]Custom Calculation Example[/bold blue]")
    console.print("-" * 40)

    # Define custom calculation
    class BiomassCoeffientOfVariation(ForestCalculation):
        """Calculate coefficient of variation across species."""

        def __init__(self, **kwargs):
            super().__init__(
                name="biomass_cv",
                description="Coefficient of variation of biomass",
                units="ratio",
                **kwargs
            )

        def calculate(self, biomass_data: np.ndarray, **kwargs) -> np.ndarray:
            """Calculate CV = std/mean for each pixel."""
            # Skip total layer (index 0)
            species_data = biomass_data[1:]

            mean_biomass = np.mean(species_data, axis=0)
            std_biomass = np.std(species_data, axis=0)

            cv = np.zeros_like(mean_biomass)
            mask = mean_biomass > 0
            cv[mask] = std_biomass[mask] / mean_biomass[mask]

            return cv

        def validate_data(self, biomass_data: np.ndarray) -> bool:
            return biomass_data.ndim == 3 and biomass_data.shape[0] > 1

    # Register custom calculation
    registry.register("biomass_cv", BiomassCoeffientOfVariation)
    console.print(f"✅ Registered custom calculation: biomass_cv")

    # Use in analysis
    zarr_path = create_sample_zarr(Path("temp_custom.zarr"), n_species=5)

    settings = GridFIASettings(
        output_dir=Path("results/custom"),
        calculations=[
            CalculationConfig(
                name="biomass_cv",
                enabled=True,
                output_format="geotiff"
            )
        ]
    )

    processor = ForestMetricsProcessor(settings)
    results = processor.run_calculations(str(zarr_path))

    console.print(f"✅ Custom calculation complete: {list(results.keys())}")

    # Clean up
    import shutil
    shutil.rmtree(zarr_path)
    if Path("results").exists():
        shutil.rmtree("results")


def example_output_formats():
    """Demonstrate different output formats."""
    console.print("\n[bold blue]Output Format Examples[/bold blue]")
    console.print("-" * 40)

    zarr_path = create_sample_zarr(Path("temp_formats.zarr"))

    # Different output formats
    formats = [
        ("geotiff", "Standard GeoTIFF for GIS"),
        ("netcdf", "NetCDF for xarray integration"),
        ("zarr", "Zarr for large-scale processing"),
    ]

    for format_type, description in formats:
        console.print(f"\n{format_type.upper()}: {description}")

        settings = GridFIASettings(
            output_dir=Path(f"results/{format_type}"),
            calculations=[
                CalculationConfig(
                    name="shannon_diversity",
                    enabled=True,
                    output_format=format_type,
                    output_name=f"diversity_{format_type}"
                )
            ]
        )

        try:
            processor = ForestMetricsProcessor(settings)
            results = processor.run_calculations(str(zarr_path))
            console.print(f"   ✅ Saved as {format_type}")
        except Exception as e:
            console.print(f"   ⚠️  {format_type} may require additional dependencies")

    # Clean up
    import shutil
    shutil.rmtree(zarr_path)
    if Path("results").exists():
        shutil.rmtree("results")


def example_calculation_parameters():
    """Show how to customize calculation parameters."""
    console.print("\n[bold blue]Calculation Parameters[/bold blue]")
    console.print("-" * 40)

    zarr_path = create_sample_zarr(Path("temp_params.zarr"))

    # Different parameter configurations
    param_examples = [
        {
            "name": "Species richness with different thresholds",
            "calculations": [
                CalculationConfig(
                    name="species_richness",
                    parameters={"biomass_threshold": 0.1},
                    output_name="richness_low_threshold"
                ),
                CalculationConfig(
                    name="species_richness",
                    parameters={"biomass_threshold": 5.0},
                    output_name="richness_high_threshold"
                )
            ]
        },
        {
            "name": "Shannon diversity with different logarithm bases",
            "calculations": [
                CalculationConfig(
                    name="shannon_diversity",
                    parameters={"base": "e"},  # Natural log
                    output_name="shannon_natural"
                ),
                CalculationConfig(
                    name="shannon_diversity",
                    parameters={"base": 2},  # Log base 2
                    output_name="shannon_base2"
                )
            ]
        }
    ]

    for example in param_examples:
        console.print(f"\n{example['name']}:")

        settings = GridFIASettings(
            output_dir=Path("results/params"),
            calculations=example["calculations"]
        )

        processor = ForestMetricsProcessor(settings)
        results = processor.run_calculations(str(zarr_path))

        for name in results:
            console.print(f"   - {name}")

    # Clean up
    import shutil
    shutil.rmtree(zarr_path)
    if Path("results").exists():
        shutil.rmtree("results")


def example_batch_calculations():
    """Run multiple calculations efficiently."""
    console.print("\n[bold blue]Batch Calculation Processing[/bold blue]")
    console.print("-" * 40)

    zarr_path = create_sample_zarr(Path("temp_batch.zarr"), n_species=10)

    # Configure comprehensive analysis
    settings = GridFIASettings(
        output_dir=Path("results/comprehensive"),
        calculations=[
            # Diversity metrics
            CalculationConfig(name="species_richness", enabled=True),
            CalculationConfig(name="shannon_diversity", enabled=True),
            CalculationConfig(name="simpson_diversity", enabled=True),
            CalculationConfig(name="evenness", enabled=True),

            # Biomass metrics
            CalculationConfig(name="total_biomass", enabled=True),
            CalculationConfig(name="dominant_species", enabled=True),

            # All with consistent output format
            CalculationConfig(
                name="species_richness",
                output_format="geotiff",
                output_name="all_metrics"
            )
        ]
    )

    console.print("Running comprehensive forest analysis...")
    processor = ForestMetricsProcessor(settings)

    # Adjust chunk size for efficiency
    processor.chunk_size = (1, 50, 50)

    results = processor.run_calculations(str(zarr_path))

    # Display results
    table = Table(title="Batch Calculation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Output File", style="yellow")

    for name, path in results.items():
        table.add_row(name, Path(path).name)

    console.print(table)

    # Clean up
    import shutil
    shutil.rmtree(zarr_path)
    if Path("results").exists():
        shutil.rmtree("results")


def main():
    """Run all calculation examples."""
    console.print("[bold green]Forest Calculations Framework[/bold green]")
    console.print("=" * 60)

    # Show available calculations
    show_available_calculations()

    # Run examples
    example_basic_calculations()
    example_custom_calculation()
    example_output_formats()
    example_calculation_parameters()
    example_batch_calculations()

    console.print("\n" + "=" * 60)
    console.print("[bold green]Calculation Examples Complete![/bold green]")
    console.print("\nKey takeaways:")
    console.print("  - Use built-in calculations for standard metrics")
    console.print("  - Create custom calculations for specific needs")
    console.print("  - Choose output format based on downstream use")
    console.print("  - Batch calculations for efficiency")
    console.print("\nSee docs/ for calculation formulas and details")


if __name__ == "__main__":
    main()