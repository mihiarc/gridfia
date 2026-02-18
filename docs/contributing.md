# Contributing

Thank you for your interest in contributing to GridFIA! This guide covers development setup, coding standards, and how to submit changes.

## Development Setup

### Clone and Install

```bash
git clone https://github.com/mihiarc/gridfia.git
cd gridfia

# Create virtual environment (using uv)
uv venv
source .venv/bin/activate

# Install in development mode with all extras
uv pip install -e ".[dev,test,docs]"
```

### Verify Installation

```bash
uv run python -c "from gridfia import GridFIA; print('OK')"
```

## Running Tests

GridFIA uses [pytest](https://docs.pytest.org/) with a minimum 80% coverage requirement.

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/unit/test_processors.py

# Run with coverage report
uv run pytest --cov

# Run tests in parallel (if pytest-xdist is installed)
uv run pytest -n auto
```

!!! note
    GridFIA tests use real API calls (no mocking). Some tests may require internet
    access and can take longer to run.

## Code Quality

### Formatting

```bash
# Format code with Black
uv run black gridfia/ tests/

# Sort imports with isort
uv run isort gridfia/ tests/
```

### Linting

```bash
# Lint with flake8
uv run flake8 gridfia/ tests/
```

### Type Checking

```bash
# Check types with mypy
uv run mypy gridfia/
```

## Documentation

### Serve Locally

```bash
uv run mkdocs serve
```

Visit `http://127.0.0.1:8000` to preview. The site auto-reloads on changes.

### Build

```bash
uv run mkdocs build --strict
```

The `--strict` flag ensures no warnings or broken links.

### Documentation Guidelines

- Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for Python code
- Include code examples in docstrings where helpful
- Use [mkdocstrings](https://mkdocstrings.github.io/) `:::` directives for auto-generated API docs
- Use [Mermaid](https://mermaid.js.org/) for architecture diagrams
- Use [MathJax](https://www.mathjax.org/) for mathematical formulas

## Adding a New Calculation

GridFIA uses a registry pattern for calculations. To add a new metric:

### 1. Create the Calculation Class

Create a new class in `gridfia/core/calculations/` inheriting from `ForestCalculation`:

```python
import numpy as np
from gridfia.core.calculations.base import ForestCalculation

class MyMetric(ForestCalculation):
    """My custom forest metric."""

    def __init__(self, **kwargs):
        super().__init__(
            name="my_metric",
            description="Description of the metric",
            units="units"
        )

    def calculate(self, biomass_data: np.ndarray, **kwargs) -> np.ndarray:
        """Calculate the metric from biomass data.

        Parameters
        ----------
        biomass_data : np.ndarray
            3D array of shape (species, height, width).

        Returns
        -------
        np.ndarray
            2D result array of shape (height, width).
        """
        # Your calculation logic here
        return result
```

### 2. Register with the Registry

```python
from gridfia.core.calculations.registry import registry

registry.register("my_metric", MyMetric)
```

### 3. Add Tests

Add tests in `tests/unit/test_calculations.py`:

```python
def test_my_metric():
    calc = registry.get("my_metric")
    biomass = np.random.rand(5, 100, 100).astype(np.float32)
    result = calc.calculate(biomass)

    assert result.shape == (100, 100)
    # Add assertions specific to your metric
```

### 4. Use via the API

```python
api = GridFIA()
results = api.calculate_metrics("data.zarr", calculations=["my_metric"])
```

## Pull Request Process

1. Fork the repository and create a feature branch
2. Make your changes with tests
3. Run the full test suite: `uv run pytest`
4. Run code quality tools: `uv run black gridfia/ tests/ && uv run isort gridfia/ tests/ && uv run flake8 gridfia/ tests/`
5. Update documentation if needed
6. Submit a pull request with a clear description

## Reporting Issues

Please report bugs and feature requests on the [GitHub issue tracker](https://github.com/mihiarc/gridfia/issues). Include:

- Python version and OS
- GridFIA version (`python -c "import gridfia; print(gridfia.__version__)"`)
- Minimal reproducible example
- Full error traceback
