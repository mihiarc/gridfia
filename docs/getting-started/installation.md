# Installation Guide

This guide will help you install GridFIA and its dependencies on your system. GridFIA supports Python 3.9+ and has been tested on Linux, macOS, and Windows.

## System Requirements

### Minimum Requirements
- **Python**: 3.9 or higher
- **Memory**: 8 GB RAM (16 GB recommended for large datasets)
- **Storage**: 5 GB free space for installation and sample data
- **Operating System**: Linux, macOS, or Windows

### Recommended System
- **Python**: 3.11 or higher
- **Memory**: 32 GB RAM or more
- **Storage**: 100 GB+ for working with full BIGMAP datasets
- **CPU**: Multi-core processor for parallel processing
- **GPU**: Optional, for accelerated computations (future releases)

## Installation Methods

### Method 1: Using uv (Recommended)

[uv](https://astral.sh/uv/) is a fast Python package manager that provides excellent dependency resolution and virtual environment management.

#### Install uv

=== "Linux/macOS"
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows"
    ```powershell
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

=== "pip"
    ```bash
    pip install uv
    ```

#### Install GridFIA with uv

```bash
# Clone the repository
git clone https://github.com/mihiarc/gridfia.git
cd gridfia

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install GridFIA in development mode
uv pip install -e .

# Install with development dependencies (optional)
uv pip install -e ".[dev,test,docs]"
```

### Method 2: Using pip

```bash
# Clone the repository
git clone https://github.com/mihiarc/gridfia.git
cd gridfia

# Create virtual environment (recommended)
python -m venv gridfia-env
source gridfia-env/bin/activate  # On Windows: gridfia-env\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install GridFIA
pip install -e .

# Install with optional dependencies
pip install -e ".[dev,test,docs]"
```

### Method 3: Using conda

```bash
# Create conda environment
conda create -n gridfia python=3.11
conda activate gridfia

# Install core dependencies from conda-forge
conda install -c conda-forge numpy pandas xarray zarr rasterio geopandas matplotlib

# Clone and install GridFIA
git clone https://github.com/mihiarc/gridfia.git
cd gridfia
pip install -e .
```

## Dependency Overview

GridFIA has several categories of dependencies:

### Core Dependencies
- **numpy** (≥1.21.0) - Scientific computing
- **pandas** (≥1.3.0) - Data analysis
- **xarray** (≥0.19.0) - N-dimensional arrays
- **zarr** (≥2.10.0) - Chunked array storage
- **rasterio** (≥1.2.0) - Geospatial raster I/O
- **geopandas** (≥0.10.0) - Geospatial data structures

### Visualization Dependencies
- **matplotlib** (≥3.4.0) - Plotting
- **rich** (≥13.0.0) - Terminal UI

### Configuration & CLI
- **pydantic** (≥2.0.0) - Data validation
- **pydantic-settings** (≥2.0.0) - Settings management
- **click** (≥8.0.0) - CLI framework
- **typer** (≥0.9.0) - Type-hinted CLI

### API & Networking
- **requests** (≥2.28.0) - HTTP client

## Verification

After installation, verify that GridFIA is working correctly:

### Test Installation

```bash
# Check GridFIA version
gridfia --version

# Test Python import
python -c "import gridfia; print(f'GridFIA v{gridfia.__version__} installed successfully!')"

# Run basic CLI help
gridfia --help
```

### Run Example Analysis

```bash
# Create test directories
mkdir -p data output

# Run a quick test (requires sample data)
gridfia-analyze --help
gridfia-visualize --help
gridfia-process --help
```

### Test Dependencies

```python
# Test script to verify all dependencies
import sys
import importlib

required_packages = [
    'numpy', 'pandas', 'xarray', 'zarr', 
    'rasterio', 'geopandas', 'matplotlib', 
    'rich', 'pydantic', 'click', 'requests'
]

missing_packages = []

for package in required_packages:
    try:
        importlib.import_module(package)
        print(f"✅ {package}")
    except ImportError:
        print(f"❌ {package}")
        missing_packages.append(package)

if missing_packages:
    print(f"\nMissing packages: {', '.join(missing_packages)}")
    sys.exit(1)
else:
    print("\n🎉 All dependencies installed successfully!")
```

## Platform-Specific Notes

### Linux

Most Linux distributions work out of the box. For Ubuntu/Debian systems, you may need:

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-dev python3-pip build-essential libgdal-dev libproj-dev
```

For RHEL/CentOS/Fedora:

```bash
# Install system dependencies
sudo yum install -y python3-devel gcc gdal-devel proj-devel
# or on newer systems:
sudo dnf install -y python3-devel gcc gdal-devel proj-devel
```

### macOS

Install Xcode command line tools if not already installed:

```bash
xcode-select --install
```

Consider using Homebrew for system dependencies:

```bash
brew install gdal proj
```

### Windows

For Windows users, we recommend using conda to handle the more complex geospatial dependencies:

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Use the conda installation method above

Alternatively, you can use the [OSGeo4W](https://trac.osgeo.org/osgeo4w/) distribution for geospatial libraries.

## Development Installation

For contributors and developers who want to modify GridFIA:

```bash
# Clone with development branch
git clone -b develop https://github.com/mihiarc/gridfia.git
cd gridfia

# Create development environment
uv venv --python 3.11
source .venv/bin/activate

# Install with all dependencies
uv pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
pre-commit install

# Run tests to verify installation
pytest
```

### Development Dependencies

Additional packages for development:

- **pytest** (≥7.0.0) - Testing framework
- **pytest-cov** (≥4.0.0) - Coverage reporting
- **black** (≥22.0.0) - Code formatting
- **isort** (≥5.10.0) - Import sorting
- **flake8** (≥5.0.0) - Linting
- **mypy** (≥1.0.0) - Type checking
- **pre-commit** (≥2.20.0) - Git hooks

## Docker Installation

For containerized deployment:

```bash
# Build Docker image
docker build -t gridfia:latest .

# Run with mounted data directory
docker run -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output gridfia:latest
```

## Troubleshooting

### Common Issues

#### GDAL/Rasterio Installation Issues

If you encounter GDAL-related errors:

1. **Linux**: Install GDAL development headers
   ```bash
   sudo apt-get install libgdal-dev  # Ubuntu/Debian
   sudo yum install gdal-devel       # RHEL/CentOS
   ```

2. **macOS**: Use conda or homebrew
   ```bash
   conda install -c conda-forge rasterio
   # or
   brew install gdal
   ```

3. **Windows**: Use conda-forge channel
   ```bash
   conda install -c conda-forge rasterio geopandas
   ```

#### Memory Issues

For large dataset processing:

1. Increase virtual memory/swap space
2. Use chunked processing options in configuration
3. Consider using a machine with more RAM

#### Permission Issues

If you encounter permission errors:

```bash
# Create virtual environment in user directory
python -m venv ~/.gridfia-env
source ~/.gridfia-env/bin/activate
pip install -e .
```

### Getting Help

If you encounter issues:

1. Check the [troubleshooting guide](../reference/troubleshooting.md)
2. Search [existing issues](https://github.com/mihiarc/gridfia/issues)
3. Create a [new issue](https://github.com/mihiarc/gridfia/issues/new) with:
   - Your operating system and Python version
   - Complete error message
   - Steps to reproduce the issue

## Next Steps

After successful installation:

1. Read the [Quick Start Guide](quickstart.md)
2. Explore the [Configuration Options](configuration.md)
3. Try the [Tutorials](../tutorials/bigmap-analysis.md)
4. Review the [User Guide](../user-guide/cli.md) 