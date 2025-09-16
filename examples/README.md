# BigMap Examples

This directory contains example scripts demonstrating BigMap functionality, organized from simple to complex.

## 📚 Example Structure

| File | Description | Time | Prerequisites |
|------|-------------|------|---------------|
| **01_quickstart.py** | Minimal working example - download, process, analyze one county | 2 min | None |
| **02_api_overview.py** | Complete API feature demonstration | 5 min | None |
| **03_location_configs.py** | Working with different geographic locations | 3 min | None |
| **04_calculations.py** | Forest calculation framework and custom metrics | 5 min | None |
| **05_species_analysis.py** | Species proportions, groups, and diversity analysis | 10 min | None |
| **06_wake_county_full.py** | Complete case study with publication outputs | 15 min | Internet connection |
| **utils.py** | Shared utilities used by all examples | - | - |

## 🚀 Getting Started

### Quick Start (2 minutes)
```bash
# Run the simplest example
python examples/01_quickstart.py
```
This downloads data for Wake County, NC and calculates species richness.

### Learning Path

1. **New Users**: Start with `01_quickstart.py`
2. **API Overview**: Run `02_api_overview.py` to see all features
3. **Specific Topics**:
   - Geographic areas → `03_location_configs.py`
   - Calculations → `04_calculations.py`
   - Species analysis → `05_species_analysis.py`
4. **Complete Workflow**: Study `06_wake_county_full.py`

## 📖 Example Details

### 01_quickstart.py
**Purpose**: Get running quickly with minimal code
- Downloads 2 species for Wake County
- Creates a Zarr store
- Calculates species richness
- Prints basic statistics

### 02_api_overview.py
**Purpose**: Demonstrate all API capabilities
- List available species
- Location configurations (state, county, custom)
- Download patterns
- Zarr operations
- Calculation configurations
- Visualization options
- Batch processing

### 03_location_configs.py
**Purpose**: Work with different geographic areas
- State-level configurations
- County-level configurations
- Custom bounding boxes
- Batch location processing
- Configuration persistence (YAML)

### 04_calculations.py
**Purpose**: Master the calculation framework
- List available calculations
- Basic diversity metrics
- Custom calculation creation
- Output format options (GeoTIFF, NetCDF, Zarr)
- Parameter customization
- Batch calculations

### 05_species_analysis.py
**Purpose**: Comprehensive species analysis
- Individual species proportions
- Species group analysis (hardwood/softwood)
- Southern Yellow Pine complex
- Diversity hotspot identification
- Statistical summaries

### 06_wake_county_full.py
**Purpose**: Complete real-world workflow
- Multi-species data download
- Zarr store creation with metadata
- All forest calculations
- Statistical analysis
- Multiple visualization types
- Publication-ready figures
- Summary report generation

## 💡 Tips

### Memory Management
Examples use chunked processing for large datasets. Adjust chunk sizes if needed:
```python
processor.chunk_size = (1, 500, 500)  # Smaller chunks for less memory
```

### Sample Data
Most examples can create sample data for testing:
```python
from examples.utils import create_sample_zarr
zarr_path = create_sample_zarr(Path("test.zarr"), n_species=5)
```

### Real Data
To work with real BIGMAP data:
1. Ensure internet connection
2. Use `BigMapAPI.download_species()`
3. Expect ~100MB per species per state

## 🛠️ Common Patterns

### Basic Workflow
```python
from bigmap import BigMapAPI

api = BigMapAPI()

# 1. Download
files = api.download_species(state="NC", county="Wake", species_codes=["0131"])

# 2. Process
zarr_path = api.create_zarr("downloads/", "data.zarr")

# 3. Analyze
results = api.calculate_metrics(zarr_path, calculations=["shannon_diversity"])

# 4. Visualize
maps = api.create_maps(zarr_path, map_type="diversity")
```

### Custom Configuration
```python
from bigmap import BigMapSettings, CalculationConfig

settings = BigMapSettings(
    output_dir=Path("custom_output"),
    calculations=[
        CalculationConfig(
            name="species_richness",
            parameters={"biomass_threshold": 2.0},
            output_format="geotiff"
        )
    ]
)

api = BigMapAPI(config=settings)
```

## 📊 Output Files

Examples create outputs in their respective directories:
- `quickstart_data/` - Downloaded data and results
- `wake_county_data/` - Wake County case study data
- `wake_results/` - Analysis outputs and figures
- `configs/` - Location configuration files
- `results/` - Calculation outputs

## 🔗 Related Resources

- **Tutorial**: See `docs/tutorials/species-diversity-analysis.md`
- **API Docs**: Run `bigmap --help` or see `docs/api/`
- **Config Examples**: Check `cfg/` directory

## ❓ Troubleshooting

### Import Errors
```bash
# Ensure BigMap is installed
pip install -e .
# or
uv pip install -e .
```

### Download Failures
- Check internet connection
- Verify species codes with `bigmap list-species`
- Some species may not be available for all locations

### Working with Custom Geographic Areas

The BigMap API supports multiple ways to specify geographic areas:

1. **Use state and county names** (Recommended):
   ```python
   files = api.download_species(
       state="North Carolina",
       county="Wake",
       species_codes=["0131", "0068"]
   )
   ```

2. **Use custom bounding box coordinates**:
   ```python
   files = api.download_species(
       bbox=(-104.5, 39.5, -104.0, 40.0),  # xmin, ymin, xmax, ymax
       crs="4326",  # WGS84
       species_codes=["0131"]
   )
   ```

   **Finding bounding boxes**: Use https://boundingbox.klokantech.com/ to visually select your area and get coordinates in different formats (WGS84, Web Mercator, etc.)

### Memory Issues
- Reduce chunk sizes
- Process smaller areas
- Use sample data for testing

### Zarr Compatibility Warnings
You may see warnings like:
```
UnstableSpecificationWarning: The data type (FixedLengthUTF32) does not have a Zarr V3 specification
```
This is expected and safe to ignore. These warnings indicate that some metadata uses data types not yet standardized in Zarr V3, but won't affect functionality.

## 📝 Contributing

To add new examples:
1. Follow the numbered naming convention
2. Include docstrings and comments
3. Use `examples.utils` for common functions
4. Keep focused on specific topics
5. Add entry to this README