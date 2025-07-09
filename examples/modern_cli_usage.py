#!/usr/bin/env python3
"""
Examples demonstrating the modern BigMap CLI following 2024 best practices.

This script shows how to use the new Typer-based CLI for various forest analysis tasks.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str = ""):
    """Run a CLI command and display results."""
    print(f"\n🔧 {description}")
    print(f"Command: {cmd}")
    print("-" * 60)
    
    try:
        result = subprocess.run(
            cmd.split(), 
            capture_output=True, 
            text=True, 
            check=True
        )
        print(result.stdout)
        if result.stderr:
            print(f"Warning: {result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
    except FileNotFoundError:
        print("❌ BigMap CLI not installed. Run 'pip install -e .' first")


def main():
    """Demonstrate modern CLI usage."""
    print("🌲 BigMap Modern CLI Examples (2024 Best Practices)")
    print("=" * 70)
    
    # Basic help and info
    print("\n📋 BASIC COMMANDS")
    run_command("bigmap --help", "Show main help")
    run_command("bigmap --version", "Show version")
    
    # Configuration management
    print("\n⚙️ CONFIGURATION MANAGEMENT")
    run_command("bigmap config show", "Show current configuration")
    run_command("bigmap config create --template diversity --output my_config.yaml", "Create diversity config")
    run_command("bigmap config validate --config my_config.yaml", "Validate configuration")
    
    # List available calculations
    print("\n🧮 CALCULATION FRAMEWORK")
    run_command("bigmap calculate data.zarr --list", "List available calculations")
    
    # API operations
    print("\n🌐 API OPERATIONS")
    run_command("bigmap list-species", "List available species from API")
    run_command("bigmap download --species 0131 --species 0068 --output downloads/", "Download specific species")
    
    # Analysis workflows
    print("\n📊 ANALYSIS WORKFLOWS")
    
    # Create sample data path for examples
    sample_zarr = "output/nc_biomass_expandable.zarr"
    sample_raster = "output/nc_species_diversity.tif"
    
    if Path(sample_zarr).exists():
        run_command(f"bigmap analyze --zarr {sample_zarr} --output results/", "Run species analysis")
        run_command(f"bigmap calculate {sample_zarr} --calc total_biomass --calc species_richness", "Calculate metrics")
    else:
        print(f"ℹ️  Sample data not found: {sample_zarr}")
        print("   Run data processing commands first")
    
    # Visualization
    if Path(sample_raster).exists():
        run_command(f"bigmap visualize {sample_raster} --type diversity --dpi 300", "Create diversity map")
    else:
        print(f"ℹ️  Sample raster not found: {sample_raster}")
    
    # Data processing
    print("\n⚙️ DATA PROCESSING")
    run_command("bigmap process clip --input rasters/ --output nc_clipped/ --help", "Show clip help")
    run_command("bigmap process zarr --input total_biomass.tif --output biomass.zarr --help", "Show zarr help")
    
    # Advanced examples
    print("\n🚀 ADVANCED EXAMPLES")
    
    # Config-driven analysis
    config_example = """
    # Create comprehensive analysis with config
    bigmap calculate data.zarr --config diversity_config.yaml --output results/diversity/
    
    # Custom calculation parameters
    bigmap calculate data.zarr --calc species_richness --output richness_maps/ \\
        --calc-param biomass_threshold=1.0
    
    # Batch processing with different configurations
    for config in configs/*.yaml; do
        bigmap calculate data.zarr --config "$config" --output "results/$(basename $config .yaml)/"
    done
    """
    
    print("Advanced usage examples:")
    print(config_example)
    
    # Integration examples
    print("\n🔗 INTEGRATION EXAMPLES")
    
    integration_examples = """
    # Python integration
    from bigmap.core.forest_metrics import run_forest_analysis
    results = run_forest_analysis('data.zarr', 'config.yaml')
    
    # Shell scripting
    #!/bin/bash
    bigmap download --species 0131 --output data/
    bigmap process zarr --input data/total_biomass.tif --output data.zarr
    bigmap calculate data.zarr --config production_config.yaml
    bigmap visualize output/total_biomass.tif --type biomass --dpi 600
    
    # CI/CD Pipeline
    - name: Run Forest Analysis
      run: |
        bigmap calculate ${{ inputs.zarr_file }} --config ${{ inputs.config }}
        bigmap visualize output/*.tif --type diversity --output reports/
    """
    
    print("Integration examples:")
    print(integration_examples)
    
    print("\n✨ Modern CLI Features Demonstrated:")
    print("  ✅ Type-safe commands with Typer")
    print("  ✅ Rich terminal output with colors and tables")
    print("  ✅ Configuration management")
    print("  ✅ Flexible calculation framework")
    print("  ✅ Comprehensive error handling")
    print("  ✅ Progress bars and status updates")
    print("  ✅ Multiple output formats")
    print("  ✅ Batch processing capabilities")


if __name__ == "__main__":
    main()