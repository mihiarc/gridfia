"""
Modern CLI for BigMap using Typer framework.

This module provides a modern, type-safe command-line interface following 2024 best practices.
Uses Typer for type hints and minimal boilerplate while maintaining Click's power.
"""

import sys
from pathlib import Path
from typing import List, Optional, Annotated
import logging

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import track

from bigmap import __version__
from bigmap.console import print_success, print_error, print_info, print_warning
from bigmap.config import BigMapSettings, load_settings, save_settings, CalculationConfig
from bigmap.core.processors.forest_metrics import ForestMetricsProcessor, run_forest_analysis
from bigmap.core.calculations import registry
from bigmap.api import BigMapRestClient

# Initialize console and app
console = Console()
app = typer.Typer(
    name="bigmap",
    help="🌲 BigMap: North Carolina Forest Biomass and Species Diversity Analysis Tools",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def version_callback(value: bool):
    """Print version and exit."""
    if value:
        print(f"BigMap v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[Optional[bool], typer.Option("--version", "-v", callback=version_callback, is_eager=True)] = None,
    verbose: Annotated[bool, typer.Option("--verbose", help="Enable verbose output")] = False,
    debug: Annotated[bool, typer.Option("--debug", help="Enable debug mode")] = False,
):
    """
    🌲 BigMap: Modern forest analysis tools for North Carolina.
    
    Analyze forest biomass, species diversity, and create publication-ready visualizations
    from BIGMAP 2018 data.
    """
    if verbose:
        logging.getLogger().setLevel(logging.INFO)
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)


@app.command("calculate")
def calculate_cmd(
    zarr_path: Annotated[Path, typer.Argument(help="Path to biomass zarr file")],
    config: Annotated[Optional[Path], typer.Option("--config", "-c", help="Configuration file path")] = None,
    output_dir: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output directory")] = None,
    calculation: Annotated[Optional[List[str]], typer.Option("--calc", help="Specific calculation to run")] = None,
    list_calculations: Annotated[bool, typer.Option("--list", help="List available calculations")] = False,
):
    """
    🧮 Calculate forest metrics using the flexible calculation framework.
    
    Examples:
        bigmap calculate data.zarr --config diversity_config.yaml
        bigmap calculate data.zarr --calc total_biomass --calc species_richness
        bigmap calculate data.zarr --list
    """
    try:
        if list_calculations:
            _list_available_calculations()
            return
        
        if not zarr_path.exists():
            print_error(f"Zarr file not found: {zarr_path}")
            raise typer.Exit(1)
        
        # Load settings
        if config and config.exists():
            settings = load_settings(config)
            print_info(f"Loaded configuration from: {config}")
        else:
            settings = BigMapSettings()
            
        # Override output directory
        if output_dir:
            settings.output_dir = output_dir
            
        # Filter calculations if specified
        if calculation:
            # Get all registered calculations
            all_registered = registry.list_calculations()
            invalid_calcs = [c for c in calculation if c not in all_registered]
            if invalid_calcs:
                print_error(f"Unknown calculations: {invalid_calcs}")
                print_info(f"Available: {all_registered}")
                raise typer.Exit(1)
            
            # Create new calculation configs for requested calculations
            settings.calculations = [
                CalculationConfig(name=calc_name, enabled=True)
                for calc_name in calculation
            ]
        
        # Run analysis
        print_info(f"🚀 Starting calculation pipeline...")
        processor = ForestMetricsProcessor(settings)
        results = processor.run_calculations(str(zarr_path))
        
        if results:
            print_success(f"✅ Completed {len(results)} calculations:")
            for name, path in results.items():
                print_info(f"  • {name} → {path}")
        else:
            print_warning("No calculations were completed")
            
    except Exception as e:
        print_error(f"Calculation failed: {e}")
        logger.exception("Detailed error:")
        raise typer.Exit(1)



@app.command("config")
def config_cmd(
    action: Annotated[str, typer.Argument(help="Action: show, create, or validate")],
    template: Annotated[Optional[str], typer.Option("--template", "-t", help="Configuration template")] = None,
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output file path")] = None,
    config_file: Annotated[Optional[Path], typer.Option("--config", "-c", help="Configuration file")] = None,
):
    """
    ⚙️ Manage BigMap configuration files.
    
    Actions:
        show      - Display current configuration
        create    - Create configuration from template
        validate  - Validate configuration file
        
    Examples:
        bigmap config show
        bigmap config create --template diversity --output my_config.yaml
        bigmap config validate --config my_config.yaml
    """
    try:
        if action == "show":
            _show_configuration(config_file)
        elif action == "create":
            _create_configuration(template, output)
        elif action == "validate":
            _validate_configuration(config_file)
        else:
            print_error(f"Unknown action: {action}")
            print_info("Available actions: show, create, validate")
            raise typer.Exit(1)
            
    except Exception as e:
        print_error(f"Configuration action failed: {e}")
        logger.exception("Detailed error:")
        raise typer.Exit(1)


# API Commands (keeping the powerful ones from the original)
@app.command("list-species")
def list_species_cmd():
    """📋 List available species from REST API."""
    try:
        print_info("🌐 Connecting to FIA BIGMAP ImageServer...")
        
        client = BigMapRestClient()
        species_list = client.list_available_species()
        
        if species_list:
            print_success(f"Found {len(species_list)} species")
            
            table = Table(title="🌲 Available Species from FIA BIGMAP REST API")
            table.add_column("Code", style="cyan", no_wrap=True)
            table.add_column("Common Name", style="green")
            table.add_column("Scientific Name", style="yellow")
            
            for species in species_list[:20]:
                table.add_row(
                    species['species_code'],
                    species['common_name'],
                    species['scientific_name']
                )
            
            console.print(table)
            
            if len(species_list) > 20:
                print_info(f"Showing first 20 of {len(species_list)} species.")
        else:
            print_error("Failed to retrieve species list")
            
    except Exception as e:
        print_error(f"Failed to list species: {e}")
        raise typer.Exit(1)


@app.command("download")
def download_cmd(
    species_codes: Annotated[Optional[List[str]], typer.Option("--species", "-s", help="Species codes to download")] = None,
    output_dir: Annotated[Path, typer.Option("--output", "-o", help="Output directory")] = Path("downloads"),
    bbox: Annotated[Optional[str], typer.Option("--bbox", "-b", help="Bounding box: 'xmin,ymin,xmax,ymax'")] = None,
):
    """
    ⬇️ Download species data via REST API.
    
    Examples:
        bigmap download --species 0131 --species 0068 --output data/
        bigmap download --bbox "-9200000,4000000,-8400000,4400000"
    """
    try:
        # Default NC species if none specified
        if not species_codes:
            species_codes = ['0131', '0068', '0132', '0110', '0316']
            print_info(f"Using default NC species: {species_codes}")
        
        # Parse bounding box
        if bbox:
            try:
                coords = [float(x.strip()) for x in bbox.split(',')]
                if len(coords) != 4:
                    raise ValueError("Bbox must have 4 coordinates")
                nc_bbox = tuple(coords)
            except ValueError as e:
                print_error(f"Invalid bbox format: {e}")
                raise typer.Exit(1)
        else:
            # Default NC bbox in Web Mercator
            nc_bbox = (-9200000, 4000000, -8400000, 4400000)
            print_info("Using default North Carolina bounding box")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        client = BigMapRestClient()
        print_info(f"📥 Downloading {len(species_codes)} species to {output_dir}")
        
        exported_files = client.batch_export_nc_species(
            nc_bbox=nc_bbox,
            output_dir=output_dir,
            species_codes=species_codes
        )
        
        if exported_files:
            print_success(f"✅ Downloaded {len(exported_files)} species rasters")
        else:
            print_error("Failed to download species data")
            raise typer.Exit(1)
            
    except Exception as e:
        print_error(f"Download failed: {e}")
        logger.exception("Detailed error:")
        raise typer.Exit(1)


# Helper functions
def _list_available_calculations():
    """Display available calculations."""
    print_info("🧮 Available Calculations:")
    
    table = Table(title="Forest Calculation Registry")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="green")
    table.add_column("Units", style="yellow")
    table.add_column("Output Type", style="blue")
    
    # Get calculation info directly from classes
    calculation_info = {
        'species_richness': ('Number of tree species per pixel', 'count', 'uint8'),
        'shannon_diversity': ('Shannon diversity index', 'index', 'float32'),
        'simpson_diversity': ('Simpson diversity index', 'index', 'float32'),
        'evenness': ('Species evenness (Pielou\'s J)', 'ratio', 'float32'),
        'total_biomass': ('Total biomass across all species', 'Mg/ha', 'float32'),
        'total_biomass_comparison': ('Total biomass difference', 'Mg/ha', 'float32'),
        'species_proportion': ('Proportion of specific species', 'ratio', 'float32'),
        'species_percentage': ('Percentage of specific species', 'percent', 'float32'),
        'species_group_proportion': ('Proportion of species group', 'ratio', 'float32'),
        'biomass_threshold': ('Areas above biomass threshold', 'binary', 'uint8'),
        'dominant_species': ('Most abundant species by biomass', 'species_id', 'int16'),
        'species_presence': ('Binary presence of species', 'binary', 'uint8'),
        'species_dominance': ('Dominance index for species', 'ratio', 'float32'),
        'rare_species': ('Count of rare species', 'count', 'uint8'),
        'common_species': ('Count of common species', 'count', 'uint8'),
    }
    
    for calc_name in registry.list_calculations():
        if calc_name in calculation_info:
            desc, units, dtype = calculation_info[calc_name]
            table.add_row(calc_name, desc, units, dtype)
    
    console.print(table)


def _show_configuration(config_file: Optional[Path] = None):
    """Show current configuration."""
    if config_file and config_file.exists():
        settings = load_settings(config_file)
        print_info(f"Configuration from: {config_file}")
    else:
        settings = BigMapSettings()
        print_info("Default configuration")
    
    table = Table(title="⚙️ BigMap Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("App Name", settings.app_name)
    table.add_row("Debug Mode", str(settings.debug))
    table.add_row("Data Directory", str(settings.data_dir))
    table.add_row("Output Directory", str(settings.output_dir))
    table.add_row("Cache Directory", str(settings.cache_dir))
    table.add_row("Enabled Calculations", str(len([c for c in settings.calculations if c.enabled])))
    
    console.print(table)


def _create_configuration(template: Optional[str], output: Optional[Path]):
    """Create configuration from template."""
    templates = {
        "basic": _create_basic_config,
        "diversity": _create_diversity_config,
        "biomass": _create_biomass_config,
    }
    
    if not template:
        print_info("Available templates: basic, diversity, biomass")
        return
        
    if template not in templates:
        print_error(f"Unknown template: {template}")
        print_info(f"Available templates: {list(templates.keys())}")
        return
    
    settings = templates[template]()
    
    if not output:
        output = Path(f"bigmap_{template}_config.yaml")
    
    save_settings(settings, output)
    print_success(f"✅ Created configuration: {output}")


def _validate_configuration(config_file: Optional[Path]):
    """Validate configuration file."""
    if not config_file or not config_file.exists():
        print_error("Configuration file not found")
        return
    
    try:
        settings = load_settings(config_file)
        print_success(f"✅ Configuration is valid: {config_file}")
        
        # Show summary
        enabled_calcs = [c.name for c in settings.calculations if c.enabled]
        print_info(f"Enabled calculations: {enabled_calcs}")
        
    except Exception as e:
        print_error(f"Invalid configuration: {e}")


def _create_basic_config() -> BigMapSettings:
    """Create basic configuration."""
    return BigMapSettings(
        calculations=[
            CalculationConfig(name="species_richness", enabled=True),
            CalculationConfig(name="total_biomass", enabled=False),
        ]
    )


def _create_diversity_config() -> BigMapSettings:
    """Create diversity analysis configuration."""
    return BigMapSettings(
        output_dir=Path("output/diversity_analysis"),
        calculations=[
            CalculationConfig(name="species_richness", enabled=True),
            CalculationConfig(name="shannon_diversity", enabled=True),
            CalculationConfig(name="total_biomass", enabled=True),
        ]
    )


def _create_biomass_config() -> BigMapSettings:
    """Create biomass analysis configuration."""
    return BigMapSettings(
        calculations=[
            CalculationConfig(name="total_biomass", enabled=True),
            CalculationConfig(name="dominant_species", enabled=True),
        ]
    )


# Direct command functions for backward compatibility
def analyze_cmd_direct():
    """Direct analyze command for console scripts."""
    # Parse minimal args and delegate to main app
    import sys
    if len(sys.argv) > 1:
        app()
    else:
        analyze_cmd()


def visualize_cmd_direct():
    """Direct visualize command for console scripts."""
    import sys
    if len(sys.argv) > 1:
        app()
    else:
        print_error("Usage: bigmap-visualize <raster_file>")


def process_cmd_direct():
    """Direct process command for console scripts."""
    import sys
    if len(sys.argv) > 1:
        app()
    else:
        print_error("Usage: bigmap-process <command> --input <path>")


def calculate_cmd_direct():
    """Direct calculate command for console scripts."""
    import sys
    if len(sys.argv) > 1:
        app()
    else:
        print_error("Usage: bigmap-calculate <zarr_file>")


def config_cmd_direct():
    """Direct config command for console scripts."""
    import sys
    if len(sys.argv) > 1:
        app()
    else:
        config_cmd("show")




if __name__ == "__main__":
    app()