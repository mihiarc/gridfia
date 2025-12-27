# System Design

This document provides detailed technical specifications for the GridFIA system architecture, explaining the design decisions, implementation patterns, and technical considerations that guide the development of this forest analysis toolkit.

## Design Philosophy

GridFIA is built on several core design principles that shape every technical decision:

### 1. **Scientific Computing First**
- Prioritize accuracy and reproducibility in all calculations
- Use established scientific computing libraries (NumPy, SciPy, Xarray)
- Maintain provenance and metadata throughout processing pipelines
- Support for peer review and validation of results

### 2. **Scalability by Design**
- Handle datasets from small research plots to state-wide analyses
- Memory-efficient processing using chunked arrays and lazy evaluation
- Configurable parallelization for different computational environments
- Streaming processing for datasets larger than available memory

### 3. **User Experience Excellence**
- Multiple interfaces (CLI, Python API, scripts) for different user types
- Rich terminal output with progress indication and helpful error messages
- Comprehensive documentation with examples and tutorials
- Intuitive command structure following common conventions

### 4. **Maintainability and Extensibility**
- Modular architecture with clear separation of concerns
- Comprehensive type hints and docstrings for all public APIs
- Extensive test coverage with automated CI/CD pipelines
- Plugin architecture for adding new analysis methods

## Core Components

### Configuration Management

The configuration system uses Pydantic for type-safe, validated settings management:

```python
class GridFIASettings(BaseSettings):
    """
    Hierarchical configuration with:
    - Environment variable support (GRIDFIA_*)
    - Type validation and conversion
    - Nested configuration objects
    - Automatic path creation
    """
    
    # Application settings
    app_name: str = "GridFIA"
    debug: bool = False
    verbose: bool = False
    
    # Path configuration
    data_dir: Path = Path("data")
    output_dir: Path = Path("output")
    cache_dir: Path = Path(".cache")
    
    # Processing configuration
    raster: RasterConfig = Field(default_factory=RasterConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
```

**Key Features:**
- **Environment Awareness**: Automatically loads settings from environment variables
- **Validation**: Type checking and custom validators ensure configuration integrity
- **Hierarchical Structure**: Nested configuration objects for different functional areas
- **Path Management**: Automatic directory creation and path resolution

### Data Storage Architecture

GridFIA uses a multi-layered data storage strategy optimized for different access patterns:

#### Primary Storage: Zarr Arrays
```python
# Zarr configuration for optimal performance
zarr_config = {
    'chunks': (1, 1000, 1000),      # Species, Height, Width
    'compression': 'lz4',           # Fast compression/decompression
    'compression_level': 5,         # Balance between size and speed
    'dtype': 'float32',             # Sufficient precision for biomass data
}
```

**Benefits:**
- **Chunked Storage**: Enables memory-efficient processing of large arrays
- **Compression**: Reduces storage requirements while maintaining fast access
- **Expandable**: Can add new species layers without rebuilding entire dataset
- **Metadata**: Rich attribute storage for data provenance and documentation

#### Secondary Storage: NetCDF/HDF5
- Used for analysis results and intermediate products
- Self-describing format with embedded metadata
- Wide tool support across scientific computing ecosystem
- Efficient for time-series and multi-dimensional analysis results

#### Tertiary Storage: GeoPackage
- Vector data storage for boundaries, points, and analysis results
- SQLite-based format with spatial indexing
- Portable single-file format ideal for sharing results
- Supports complex geometries and attribute tables

### Processing Pipeline Design

The data processing pipeline implements a robust ETL (Extract, Transform, Load) pattern:

```python
class ProcessingPipeline:
    """
    Configurable processing pipeline with:
    - Stage validation and error handling
    - Progress tracking and logging
    - Parallel processing support
    - Checkpoint/resume capability
    """
    
    def __init__(self, config: GridFIASettings):
        self.config = config
        self.logger = self._setup_logging()
        self.progress = self._create_progress_tracker()
    
    def process(self, input_data: Path) -> ProcessingResult:
        """Execute full processing pipeline with error recovery."""
        try:
            # 1. Validation stage
            validated_data = self.validate_input(input_data)
            
            # 2. Transformation stages
            clipped_data = self.clip_to_boundary(validated_data)
            zarr_data = self.convert_to_zarr(clipped_data)
            
            # 3. Quality assurance
            qa_result = self.quality_check(zarr_data)
            
            # 4. Metadata generation
            metadata = self.generate_metadata(zarr_data, qa_result)
            
            return ProcessingResult(data=zarr_data, metadata=metadata)
            
        except ProcessingError as e:
            self.logger.error(f"Processing failed: {e}")
            return self.handle_error(e)
```

#### Key Processing Stages

1. **Input Validation**
   - Verify file formats and spatial reference systems
   - Check data integrity and completeness
   - Validate against expected schemas and ranges

2. **Spatial Operations**
   - Coordinate system transformations using PROJ
   - Clipping and masking operations with proper handling of edge cases
   - Resampling and alignment to common grids

3. **Data Transformation**
   - Format conversion (GeoTIFF → Zarr, NetCDF)
   - Compression and chunking optimization
   - Metadata preservation and enhancement

4. **Quality Assurance**
   - Statistical validation of results
   - Spatial integrity checks
   - Comparison with reference datasets where available

### Analysis Engine Architecture

The analysis engine uses a modular, plugin-based architecture:

```python
class AnalysisEngine:
    """
    Pluggable analysis engine supporting:
    - Multiple analysis types
    - Configurable parameters
    - Result caching and persistence
    - Parallel execution
    """
    
    def __init__(self, config: GridFIASettings):
        self.config = config
        self.analyzers = self._load_analyzers()
        self.cache = self._setup_cache()
    
    def register_analyzer(self, analyzer_class: Type[BaseAnalyzer]):
        """Register new analysis methods dynamically."""
        self.analyzers[analyzer_class.name] = analyzer_class
    
    def analyze(self, data: xr.Dataset, method: str, **kwargs) -> AnalysisResult:
        """Execute analysis with caching and error handling."""
        if method not in self.analyzers:
            raise ValueError(f"Unknown analysis method: {method}")
        
        # Check cache for existing results
        cache_key = self._generate_cache_key(data, method, kwargs)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Execute analysis
        analyzer = self.analyzers[method](self.config)
        result = analyzer.analyze(data, **kwargs)
        
        # Cache result for future use
        self.cache[cache_key] = result
        return result
```

#### Built-in Analysis Methods

1. **Species Presence Analysis**
   - Binary presence/absence mapping
   - Abundance threshold analysis
   - Spatial distribution patterns

2. **Diversity Metrics**
   - Shannon diversity index
   - Simpson diversity index
   - Species richness calculations
   - Evenness measures

3. **Spatial Statistics**
   - Spatial autocorrelation analysis
   - Hotspot detection (Getis-Ord Gi*)
   - Landscape connectivity metrics

4. **Temporal Analysis**
   - Trend detection and significance testing
   - Change point analysis
   - Seasonal decomposition

### Visualization System

The visualization system prioritizes publication-quality output while maintaining flexibility:

```python
class VisualizationEngine:
    """
    Publication-quality visualization with:
    - Consistent styling and branding
    - Multiple output formats
    - Interactive and static options
    - Customizable themes
    """
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.themes = self._load_themes()
        self.style_manager = StyleManager(config)
    
    def create_map(self, data: xr.DataArray, map_type: str, **kwargs) -> Figure:
        """Create publication-ready maps with consistent styling."""
        # Apply theme and styling
        style = self.style_manager.get_style(map_type)
        
        # Create base map
        fig, ax = plt.subplots(figsize=style.figure_size, dpi=style.dpi)
        
        # Add data layer with appropriate colormap
        im = data.plot(ax=ax, cmap=style.colormap, **style.plot_kwargs)
        
        # Add cartographic elements
        self._add_north_arrow(ax)
        self._add_scale_bar(ax)
        self._add_legend(im, style.legend_config)
        
        # Apply final styling
        self._apply_layout(fig, ax, style)
        
        return fig
```

#### Visualization Features

1. **Cartographic Standards**
   - Proper coordinate system labeling
   - Scale bars and north arrows
   - Professional typography and layout

2. **Color Theory Application**
   - Colorblind-friendly palettes
   - Perceptually uniform color spaces
   - Appropriate color schemes for data types

3. **Interactive Elements**
   - Hover tooltips with data values
   - Zoom and pan functionality
   - Layer toggling and transparency controls

4. **Export Options**
   - Multiple formats (PNG, PDF, SVG)
   - Configurable resolution and quality
   - Embedded metadata for reproducibility

## API Design Patterns

### REST Client Architecture

The REST API client implements robust patterns for external service integration:

```python
class BigMapRestClient:  # Note: This class name references the USDA BIGMAP data source
    """
    Production-ready REST client with:
    - Automatic retry with exponential backoff
    - Rate limiting and request throttling
    - Session management and connection pooling
    - Comprehensive error handling
    """
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.session = self._create_session()
        self.rate_limiter = RateLimiter(config.rate_limit)
        
    def _create_session(self) -> requests.Session:
        """Create configured session with retry strategy."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    @retry_on_failure
    @rate_limited
    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make rate-limited request with comprehensive error handling."""
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            self._handle_request_error(e)
            raise
```

### CLI Design Patterns

The command-line interface follows Unix philosophy and modern CLI best practices:

```python
@click.group()
@click.version_option(version=__version__)
@click.option('--config', type=click.Path(), help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def gridfia_cli(ctx, config, verbose):
    """GridFIA: Forest Analysis Toolkit."""
    ctx.ensure_object(dict)
    ctx.obj['config'] = load_config(config) if config else GridFIASettings()
    ctx.obj['verbose'] = verbose

@gridfia_cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=True)
@click.option('--output', '-o', type=click.Path(), required=True)
@click.option('--method', type=click.Choice(['shannon', 'simpson', 'richness']))
@click.pass_context
def analyze(ctx, input, output, method):
    """Run species diversity analysis."""
    config = ctx.obj['config']
    
    with Progress() as progress:
        task = progress.add_task("Analyzing...", total=100)
        
        # Execute analysis with progress updates
        result = run_analysis(input, output, method, progress_callback=progress.update)
        
        console.print(f"✅ Analysis complete: {result.summary}")
```

## Performance Optimization Strategies

### Memory Management

1. **Chunked Processing**
   ```python
   # Process data in chunks to manage memory usage
   chunk_size = calculate_optimal_chunk_size(available_memory, data_shape)
   
   for chunk in data.chunks(chunk_size):
       result_chunk = process_chunk(chunk)
       write_chunk_to_output(result_chunk)
   ```

2. **Lazy Evaluation**
   ```python
   # Use Xarray's lazy evaluation for memory efficiency
   dataset = xr.open_dataset('large_file.nc', chunks={'time': 10})
   result = dataset.groupby('time.season').mean()  # Lazy operation
   computed_result = result.compute()  # Trigger computation
   ```

3. **Memory Monitoring**
   ```python
   def monitor_memory_usage(func):
       """Decorator to monitor memory usage during processing."""
       def wrapper(*args, **kwargs):
           initial_memory = psutil.Process().memory_info().rss
           result = func(*args, **kwargs)
           final_memory = psutil.Process().memory_info().rss
           
           memory_delta = (final_memory - initial_memory) / 1024**2  # MB
           logger.info(f"Memory usage: {memory_delta:.1f} MB")
           
           return result
       return wrapper
   ```

### Computational Optimization

1. **Vectorized Operations**
   ```python
   # Use NumPy vectorization instead of loops
   # Slow: loop-based calculation
   result = np.zeros_like(data)
   for i in range(data.shape[0]):
       for j in range(data.shape[1]):
           result[i, j] = calculate_diversity(data[i, j])
   
   # Fast: vectorized calculation
   result = np.vectorize(calculate_diversity)(data)
   ```

2. **Parallel Processing**
   ```python
   from concurrent.futures import ProcessPoolExecutor
   
   def parallel_analysis(data_chunks, analysis_func):
       """Process data chunks in parallel."""
       with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
           futures = [executor.submit(analysis_func, chunk) for chunk in data_chunks]
           results = [future.result() for future in futures]
       return combine_results(results)
   ```

3. **Caching Strategy**
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=128)
   def expensive_calculation(data_hash, method):
       """Cache expensive calculations using content hash."""
       return perform_calculation(data_hash, method)
   ```

## Security Considerations

### Input Validation

```python
def validate_raster_input(file_path: Path) -> None:
    """Comprehensive raster file validation."""
    # File existence and permissions
    if not file_path.exists():
        raise ValidationError(f"File not found: {file_path}")
    
    if not os.access(file_path, os.R_OK):
        raise ValidationError(f"File not readable: {file_path}")
    
    # File format validation
    try:
        with rasterio.open(file_path) as dataset:
            # Check for valid spatial reference
            if dataset.crs is None:
                raise ValidationError("Raster has no spatial reference system")
            
            # Validate data type and ranges
            if dataset.dtypes[0] not in ['float32', 'float64', 'int16', 'int32']:
                raise ValidationError(f"Unsupported data type: {dataset.dtypes[0]}")
                
    except rasterio.errors.RasterioIOError as e:
        raise ValidationError(f"Invalid raster file: {e}")
```

### Safe External API Usage

```python
def safe_api_request(url: str, params: dict) -> dict:
    """Make safe API requests with input sanitization."""
    # URL validation
    parsed_url = urlparse(url)
    if parsed_url.scheme not in ['http', 'https']:
        raise SecurityError("Only HTTP/HTTPS URLs allowed")
    
    # Parameter sanitization
    safe_params = {}
    for key, value in params.items():
        if not isinstance(key, str) or not key.isalnum():
            raise SecurityError(f"Invalid parameter name: {key}")
        safe_params[key] = str(value)[:1000]  # Limit parameter length
    
    # Make request with timeout
    response = requests.get(url, params=safe_params, timeout=30)
    response.raise_for_status()
    
    return response.json()
```

## Testing Strategy

### Unit Testing Framework

```python
import pytest
from unittest.mock import Mock, patch
from gridfia.core import analyze_species_presence

class TestSpeciesAnalysis:
    """Comprehensive test suite for species analysis."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return create_test_zarr_array()
    
    def test_species_presence_calculation(self, sample_data):
        """Test basic species presence calculation."""
        result = analyze_species_presence(sample_data)
        
        assert result.shape == (10, 10)  # Expected output shape
        assert 0 <= result.min() <= result.max() <= 1  # Valid range
        assert not np.isnan(result).any()  # No NaN values
    
    @patch('gridfia.utils.zarr.open')
    def test_file_not_found_handling(self, mock_zarr_open):
        """Test graceful handling of missing files."""
        mock_zarr_open.side_effect = FileNotFoundError()
        
        with pytest.raises(FileNotFoundError):
            analyze_species_presence('nonexistent.zarr')
    
    def test_edge_cases(self, sample_data):
        """Test edge cases and boundary conditions."""
        # Test with empty data
        empty_data = np.zeros_like(sample_data)
        result = analyze_species_presence(empty_data)
        assert result.sum() == 0
        
        # Test with single species
        single_species = sample_data[:1]
        result = analyze_species_presence(single_species)
        assert result.shape[0] == 1
```

### Integration Testing

```python
class TestDataPipeline:
    """Integration tests for complete data processing pipeline."""
    
    def test_full_pipeline(self, tmp_path):
        """Test complete pipeline from raw data to analysis results."""
        # Setup test data
        input_file = create_test_geotiff(tmp_path / "input.tif")
        output_dir = tmp_path / "output"
        
        # Run pipeline
        pipeline = ProcessingPipeline(test_config)
        result = pipeline.process(input_file, output_dir)
        
        # Validate results
        assert result.success
        assert (output_dir / "processed.zarr").exists()
        assert result.metadata['species_count'] > 0
        
        # Validate output data integrity
        with zarr.open(output_dir / "processed.zarr") as arr:
            assert arr.shape == expected_shape
            assert arr.attrs['processing_date'] is not None
```

This comprehensive system design ensures GridFIA provides a robust, scalable, and maintainable platform for forest analysis while following software engineering best practices and scientific computing standards. 