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
from bigmap.utils.zarr_utils import create_zarr_from_geotiffs, validate_zarr_store
from bigmap.visualization.mapper import ZarrMapper
from bigmap.visualization.plots import set_plot_style, save_figure
import matplotlib.pyplot as plt

# Initialize console and app
console = Console()
app = typer.Typer(
    name="bigmap",
    help="🌲 BigMap: Forest Biomass and Species Diversity Analysis Tools",
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
    🌲 BigMap: Modern forest analysis tools for any US location.
    
    Analyze forest biomass, species diversity, and create publication-ready visualizations
    from BIGMAP 2018 data for any state, county, or custom region.
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


@app.command("location")
def location_cmd(
    action: Annotated[str, typer.Argument(help="Action: create, show, list")],
    state: Annotated[Optional[str], typer.Option("--state", "-s", help="State name or abbreviation")] = None,
    county: Annotated[Optional[str], typer.Option("--county", "-c", help="County name (requires --state)")] = None,
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output configuration file")] = None,
    bbox: Annotated[Optional[str], typer.Option("--bbox", "-b", help="Custom bbox: 'xmin,ymin,xmax,ymax'")] = None,
    crs: Annotated[str, typer.Option("--crs", help="CRS for custom bbox")] = "EPSG:4326",
):
    """
    📍 Manage location configurations for any US state or county.
    
    Actions:
        create - Create configuration for a location
        show   - Show configuration details
        list   - List available states
        
    Examples:
        bigmap location create --state NC --output north_carolina.yaml
        bigmap location create --state TX --county Harris --output houston.yaml
        bigmap location create --bbox "-104,44,-104.5,44.5" --output custom.yaml
        bigmap location show --state Montana
        bigmap location list
    """
    try:
        from bigmap.utils.location_config import LocationConfig
        from bigmap.visualization.boundaries import STATE_ABBR
        
        if action == "list":
            # List all available states
            table = Table(title="📍 Available US States")
            table.add_column("State", style="cyan")
            table.add_column("Abbreviation", style="green")
            
            for state_name, abbr in sorted(STATE_ABBR.items()):
                table.add_row(state_name.title(), abbr)
            
            console.print(table)
            
        elif action == "create":
            if county and not state:
                print_error("County requires --state to be specified")
                raise typer.Exit(1)
                
            if bbox:
                # Create from custom bbox
                try:
                    coords = [float(x.strip()) for x in bbox.split(',')]
                    if len(coords) != 4:
                        raise ValueError("Bbox must have 4 coordinates")
                    bbox_tuple = tuple(coords)
                    
                    config = LocationConfig.from_bbox(
                        bbox_tuple, 
                        name="Custom Region",
                        crs=crs,
                        output_path=output
                    )
                    print_success(f"✅ Created custom location configuration")
                    
                except ValueError as e:
                    print_error(f"Invalid bbox format: {e}")
                    raise typer.Exit(1)
                    
            elif county:
                config = LocationConfig.from_county(county, state, output_path=output)
                print_success(f"✅ Created configuration for {county} County, {state}")
                
            elif state:
                config = LocationConfig.from_state(state, output_path=output)
                print_success(f"✅ Created configuration for {state}")
                
            else:
                print_error("Please specify --state, --county, or --bbox")
                raise typer.Exit(1)
                
            config.print_summary()
            
        elif action == "show":
            if not state and not output:
                print_error("Please specify --state or provide a config file with --output")
                raise typer.Exit(1)
                
            if output and output.exists():
                config = LocationConfig(output)
            elif state:
                config = LocationConfig.from_state(state)
            
            config.print_summary()
            
        else:
            print_error(f"Unknown action: {action}")
            print_info("Available actions: create, show, list")
            raise typer.Exit(1)
            
    except Exception as e:
        print_error(f"Location command failed: {e}")
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
    state: Annotated[Optional[str], typer.Option("--state", help="State name or abbreviation")] = None,
    county: Annotated[Optional[str], typer.Option("--county", help="County name (requires --state)")] = None,
    crs: Annotated[str, typer.Option("--crs", help="CRS for bbox (default: Web Mercator)")] = "102100",
    location_config: Annotated[Optional[Path], typer.Option("--location-config", "-l", help="Location configuration file")] = None,
):
    """
    ⬇️ Download species data via REST API for any location.
    
    Examples:
        bigmap download --state NC --species 0131 --species 0068
        bigmap download --state Montana --county Missoula
        bigmap download --bbox "-9200000,4000000,-8400000,4400000"
        bigmap download --location-config texas_config.yaml
    """
    try:
        from bigmap.utils.location_config import LocationConfig
        
        # Determine location and bbox
        location_name = "location"
        location_bbox = None
        bbox_crs = crs
        
        if location_config and location_config.exists():
            # Load from config file
            config = LocationConfig(location_config)
            location_name = config.location_name.lower().replace(' ', '_')
            location_bbox = config.web_mercator_bbox
            print_info(f"Using location config: {config.location_name}")
            
        elif state:
            # Create config for state/county
            if county:
                config = LocationConfig.from_county(county, state)
                location_name = f"{county}_{state}".lower().replace(' ', '_')
            else:
                config = LocationConfig.from_state(state)
                location_name = state.lower().replace(' ', '_')
            
            location_bbox = config.web_mercator_bbox
            print_info(f"Using {config.location_name} boundaries")
            
        elif bbox:
            # Parse manual bbox
            try:
                coords = [float(x.strip()) for x in bbox.split(',')]
                if len(coords) != 4:
                    raise ValueError("Bbox must have 4 coordinates")
                location_bbox = tuple(coords)
            except ValueError as e:
                print_error(f"Invalid bbox format: {e}")
                raise typer.Exit(1)
        else:
            print_error("Please specify --state, --bbox, or --location-config")
            raise typer.Exit(1)
        
        if not location_bbox:
            print_error("Could not determine bounding box for location")
            raise typer.Exit(1)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        client = BigMapRestClient()
        print_info(f"📥 Downloading species for {location_name} to {output_dir}")
        
        exported_files = client.batch_export_location_species(
            bbox=location_bbox,
            output_dir=output_dir,
            species_codes=species_codes,
            location_name=location_name,
            bbox_srs=bbox_crs
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


@app.command("build-zarr")
def build_zarr_cmd(
    data_dir: Annotated[Path, typer.Option("--data-dir", "-d", help="Directory containing GeoTIFF files")] = Path("downloads"),
    output: Annotated[Path, typer.Option("--output", "-o", help="Output path for Zarr store")] = Path("data/forest.zarr"),
    species_codes: Annotated[Optional[List[str]], typer.Option("--species", "-s", help="Species codes to include")] = None,
    chunk_size: Annotated[str, typer.Option("--chunk-size", help="Chunk dimensions 'species,height,width'")] = "1,1000,1000",
    compression: Annotated[str, typer.Option("--compression", help="Compression algorithm")] = "lz4",
    compression_level: Annotated[int, typer.Option("--compression-level", help="Compression level (1-9)")] = 5,
    include_total: Annotated[bool, typer.Option("--include-total/--calculate-total", help="Include or calculate total biomass")] = True,
):
    """
    🏗️ Build a Zarr store from downloaded GeoTIFF files.
    
    Examples:
        bigmap build-zarr --data-dir data/montana_species --output data/montana.zarr
        bigmap build-zarr --species 0202 --species 0122 --species 0116
        bigmap build-zarr --chunk-size "1,2000,2000" --compression zstd
    """
    try:
        if not data_dir.exists():
            print_error(f"Data directory does not exist: {data_dir}")
            raise typer.Exit(1)
        
        # Parse chunk size
        try:
            chunks = tuple(int(x.strip()) for x in chunk_size.split(','))
            if len(chunks) != 3:
                raise ValueError("Chunk size must have 3 dimensions")
        except ValueError as e:
            print_error(f"Invalid chunk size format: {e}")
            raise typer.Exit(1)
        
        # Find GeoTIFF files
        tiff_files = list(data_dir.glob("*.tif")) + list(data_dir.glob("*.tiff"))
        
        if not tiff_files:
            print_error(f"No GeoTIFF files found in {data_dir}")
            raise typer.Exit(1)
        
        print_info(f"Found {len(tiff_files)} GeoTIFF files")
        
        # Filter by species codes if provided
        if species_codes:
            filtered_files = []
            for f in tiff_files:
                for code in species_codes:
                    if code in f.name:
                        filtered_files.append(f)
                        break
            tiff_files = filtered_files
            
            if not tiff_files:
                print_error(f"No files found for species codes: {species_codes}")
                raise typer.Exit(1)
        
        # Sort files for consistent ordering
        tiff_files.sort()
        
        # Extract species information from filenames
        file_species_codes = []
        file_species_names = []
        
        for f in tiff_files:
            # Try to extract species code from filename
            # Expected formats: montana_0202_Douglas-fir.tif, 0202_douglas_fir.tif, etc.
            filename = f.stem
            code = None
            name = filename
            
            # Look for 4-digit species code
            import re
            match = re.search(r'(\d{4})', filename)
            if match:
                code = match.group(1)
                # Try to extract name after code
                parts = filename.split(code)
                if len(parts) > 1:
                    name = parts[1].strip('_- ').replace('_', ' ')
            
            file_species_codes.append(code or filename[:4])
            file_species_names.append(name.title())
        
        # Display files to process
        table = Table(title="Files to Process")
        table.add_column("File", style="yellow")
        table.add_column("Code", style="cyan")
        table.add_column("Species", style="green")
        
        for f, code, name in zip(tiff_files, file_species_codes, file_species_names):
            table.add_row(f.name, code, name)
        
        console.print(table)
        
        print_info(f"Building Zarr store at: {output}")
        print_info(f"Chunk size: {chunks}")
        print_info(f"Compression: {compression} (level {compression_level})")
        
        # Create the Zarr store
        create_zarr_from_geotiffs(
            output_zarr_path=output,
            geotiff_paths=tiff_files,
            species_codes=file_species_codes,
            species_names=file_species_names,
            chunk_size=chunks,
            compression=compression,
            compression_level=compression_level,
            include_total=include_total
        )
        
        # Validate the created store
        info = validate_zarr_store(output)
        
        print_success(f"✅ Created Zarr store: {output}")
        print_info(f"Shape: {info['shape']}")
        print_info(f"Species: {info['num_species']}")
        
    except Exception as e:
        print_error(f"Failed to build Zarr store: {e}")
        logger.exception("Detailed error:")
        raise typer.Exit(1)


@app.command("map")
def map_cmd(
    zarr_path: Annotated[Path, typer.Argument(help="Path to Zarr store")],
    map_type: Annotated[str, typer.Option("--type", "-t", help="Map type: species, diversity, richness, comparison")] = "species",
    species: Annotated[Optional[List[str]], typer.Option("--species", "-s", help="Species codes for species/comparison maps")] = None,
    output_dir: Annotated[Path, typer.Option("--output", "-o", help="Output directory")] = Path("maps"),
    format: Annotated[str, typer.Option("--format", "-f", help="Output format")] = "png",
    dpi: Annotated[int, typer.Option("--dpi", help="Output resolution")] = 300,
    cmap: Annotated[Optional[str], typer.Option("--cmap", help="Colormap name")] = None,
    style: Annotated[str, typer.Option("--style", help="Plot style")] = "publication",
    show_all: Annotated[bool, typer.Option("--all", help="Create maps for all species")] = False,
    state: Annotated[Optional[str], typer.Option("--state", help="State boundary to overlay (name or abbreviation)")] = None,
    basemap: Annotated[Optional[str], typer.Option("--basemap", help="Basemap provider (OpenStreetMap, CartoDB, ESRI)")] = None,
):
    """
    🗺️ Create maps from Zarr stores.
    
    Examples:
        bigmap map data/forest.zarr --type species --species 0202
        bigmap map data/forest.zarr --type diversity --output diversity_maps/
        bigmap map data/forest.zarr --type comparison --species 0202 --species 0122
        bigmap map data/forest.zarr --all --output all_species_maps/
        bigmap map data/montana.zarr --type species --species 0202 --state MT --basemap CartoDB
    """
    try:
        if not zarr_path.exists():
            print_error(f"Zarr store not found: {zarr_path}")
            raise typer.Exit(1)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plot style
        set_plot_style(style)
        
        # Initialize mapper
        print_info(f"Loading Zarr store: {zarr_path}")
        mapper = ZarrMapper(zarr_path)
        
        # Get default colormap if not specified
        if cmap is None:
            cmap_defaults = {
                'species': 'viridis',
                'diversity': 'plasma',
                'richness': 'Spectral_r',
                'comparison': 'viridis'
            }
            cmap = cmap_defaults.get(map_type, 'viridis')
        
        if map_type == "species":
            # Create species maps
            if show_all:
                # Create maps for all species
                species_info = mapper.get_species_info()
                print_info(f"Creating maps for {len(species_info)} species...")
                
                for sp in track(species_info, description="Creating species maps"):
                    if sp['code'] != '0000':  # Skip total biomass
                        fig, ax = mapper.create_species_map(
                            species=sp['index'],
                            cmap=cmap,
                            state_boundary=state,
                            basemap=basemap
                        )
                        
                        output_path = output_dir / f"species_{sp['code']}_{sp['name'].replace(' ', '_')}.{format}"
                        save_figure(fig, str(output_path), dpi=dpi)
                        plt.close(fig)
                
                print_success(f"Created {len(species_info)} species maps")
            
            elif species:
                # Create maps for specified species
                for sp_code in species:
                    print_info(f"Creating map for species {sp_code}")
                    fig, ax = mapper.create_species_map(
                        species=sp_code,
                        cmap=cmap,
                        state_boundary=state,
                        basemap=basemap
                    )
                    
                    output_path = output_dir / f"species_{sp_code}.{format}"
                    save_figure(fig, str(output_path), dpi=dpi)
                    print_success(f"Saved: {output_path}")
                    plt.close(fig)
            else:
                print_error("Please specify species codes with --species or use --all")
                raise typer.Exit(1)
        
        elif map_type == "diversity":
            # Create diversity maps
            for div_type in ['shannon', 'simpson']:
                print_info(f"Creating {div_type} diversity map")
                fig, ax = mapper.create_diversity_map(
                    diversity_type=div_type,
                    cmap=cmap,
                    state_boundary=state,
                    basemap=basemap
                )
                
                output_path = output_dir / f"{div_type}_diversity.{format}"
                save_figure(fig, str(output_path), dpi=dpi)
                print_success(f"Saved: {output_path}")
                plt.close(fig)
        
        elif map_type == "richness":
            # Create richness map
            print_info("Creating species richness map")
            fig, ax = mapper.create_richness_map(
                cmap=cmap,
                state_boundary=state,
                basemap=basemap
            )
            
            output_path = output_dir / f"species_richness.{format}"
            save_figure(fig, str(output_path), dpi=dpi)
            print_success(f"Saved: {output_path}")
            plt.close(fig)
        
        elif map_type == "comparison":
            # Create comparison map
            if not species or len(species) < 2:
                print_error("Comparison maps require at least 2 species (use --species multiple times)")
                raise typer.Exit(1)
            
            print_info(f"Creating comparison map for {len(species)} species")
            fig = mapper.create_comparison_map(
                species_list=species,
                cmap=cmap
            )
            
            output_path = output_dir / f"species_comparison.{format}"
            save_figure(fig, str(output_path), dpi=dpi)
            print_success(f"Saved: {output_path}")
            plt.close(fig)
        
        else:
            print_error(f"Unknown map type: {map_type}")
            print_info("Valid types: species, diversity, richness, comparison")
            raise typer.Exit(1)
        
        print_success(f"✅ Maps created in {output_dir}")
        
    except Exception as e:
        print_error(f"Failed to create maps: {e}")
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