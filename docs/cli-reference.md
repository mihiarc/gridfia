# CLI Reference

The GridFIA CLI provides commands for forest analysis, data management, and configuration.

## Global Options

```bash
gridfia [OPTIONS] COMMAND [ARGS]...
```

**Options:**
- `--version`, `-v`: Show version and exit
- `--verbose`: Enable verbose output
- `--debug`: Enable debug mode
- `--help`: Show help message

## Commands

### calculate

Calculate forest metrics using the flexible calculation framework.

```bash
gridfia calculate ZARR_PATH [OPTIONS]
```

**Arguments:**
- `ZARR_PATH`: Path to biomass zarr file (required)

**Options:**
- `--config`, `-c PATH`: Configuration file path
- `--output`, `-o PATH`: Output directory
- `--calc TEXT`: Specific calculation to run (can be used multiple times)
- `--list`: List available calculations

**Examples:**

```bash
# List available calculations
gridfia calculate data.zarr --list

# Run specific calculations
gridfia calculate data.zarr --calc total_biomass --calc species_richness

# Use configuration file
gridfia calculate data.zarr --config diversity_config.yaml

# Custom output directory
gridfia calculate data.zarr --calc shannon_diversity --output results/
```

### config

Manage GridFIA configuration files.

```bash
gridfia config ACTION [OPTIONS]
```

**Actions:**
- `show`: Display current configuration
- `create`: Create configuration from template
- `validate`: Validate configuration file

**Options:**
- `--template`, `-t TEXT`: Configuration template (for create action)
- `--output`, `-o PATH`: Output file path (for create action)
- `--config`, `-c PATH`: Configuration file to validate/show

**Examples:**

```bash
# Show default configuration
gridfia config show

# Show specific configuration
gridfia config show --config my_config.yaml

# Create diversity analysis configuration
gridfia config create --template diversity --output diversity_config.yaml

# Validate configuration
gridfia config validate --config my_config.yaml
```

**Available Templates:**
- `basic`: Basic configuration with essential calculations
- `diversity`: Diversity analysis (richness, Shannon, Simpson)
- `biomass`: Biomass analysis (total, dominant species)

### list-species

List available species from the FIA BIGMAP REST API.

```bash
gridfia list-species
```

**Example Output:**
```
🌐 Connecting to FIA BIGMAP ImageServer...
✅ Found 324 species

🌲 Available Species from FIA BIGMAP REST API
┏━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Code  ┃ Common Name       ┃ Scientific Name       ┃
┡━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│ 0012  │ Balsam fir        │ Abies balsamea        │
│ 0015  │ White fir         │ Abies concolor        │
│ ...   │ ...               │ ...                   │
└───────┴───────────────────┴───────────────────────┘
```

### download

Download species data via REST API.

```bash
gridfia download [OPTIONS]
```

**Options:**
- `--species`, `-s TEXT`: Species codes to download (can be used multiple times)
- `--output`, `-o PATH`: Output directory (default: "downloads")
- `--bbox`, `-b TEXT`: Bounding box as 'xmin,ymin,xmax,ymax'

**Examples:**

```bash
# Download default NC species
gridfia download --output data/

# Download specific species
gridfia download --species 0131 --species 0068 --output data/

# Download with custom bounding box
gridfia download --bbox "-9200000,4000000,-8400000,4400000" --output data/

# Download multiple species
gridfia download \
    --species 0131 \  # Loblolly pine
    --species 0068 \  # Eastern white pine
    --species 0110 \  # Shortleaf pine
    --output nc_pines/
```

**Default Species (NC):**
- 0131: Loblolly pine
- 0068: Eastern white pine
- 0132: Longleaf pine
- 0110: Shortleaf pine
- 0316: Eastern redcedar

## Available Calculations

The following calculations are available in the registry:

| Name | Description | Units | Output Type |
|------|-------------|-------|-------------|
| `biomass_threshold` | Areas above biomass threshold | binary | uint8 |
| `common_species` | Count of common species | count | uint8 |
| `dominant_species` | Most abundant species by biomass | species_id | int16 |
| `evenness` | Species evenness (Pielou's J) | ratio | float32 |
| `rare_species` | Count of rare species | count | uint8 |
| `shannon_diversity` | Shannon diversity index | index | float32 |
| `simpson_diversity` | Simpson diversity index | index | float32 |
| `species_dominance` | Dominance index for species | ratio | float32 |
| `species_group_proportion` | Proportion of species group | ratio | float32 |
| `species_percentage` | Percentage of specific species | percent | float32 |
| `species_presence` | Binary presence of species | binary | uint8 |
| `species_proportion` | Proportion of specific species | ratio | float32 |
| `species_richness` | Number of tree species per pixel | count | uint8 |
| `total_biomass` | Total biomass across all species | Mg/ha | float32 |
| `total_biomass_comparison` | Total biomass difference | Mg/ha | float32 |

## Environment Variables

GridFIA settings can be configured via environment variables:

```bash
# Enable debug mode
export GRIDFIA_DEBUG=true

# Set output directory
export GRIDFIA_OUTPUT_DIR=/path/to/output

# Set data directory
export GRIDFIA_DATA_DIR=/path/to/data

# Enable verbose output
export GRIDFIA_VERBOSE=true
```

## Exit Codes

- `0`: Success
- `1`: General error (invalid arguments, failed calculations, etc.)
- `2`: File not found
- `3`: Invalid configuration

## Tips and Best Practices

1. **Use Configuration Files**: For complex analyses, create a configuration file
2. **Check Available Calculations**: Use `--list` to see what's available
3. **Batch Downloads**: Download multiple species at once to minimize API calls
4. **Custom Bounding Boxes**: Use smaller areas for testing before full analysis
5. **Output Formats**: Choose appropriate formats (GeoTIFF for GIS, NetCDF for xarray, Zarr for large data)