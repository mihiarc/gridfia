# Contributing to GridFIA

Thank you for your interest in contributing to GridFIA! This document provides guidelines and instructions for contributing.

## Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager (recommended)
- Git

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/gridfia.git
   cd gridfia
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e ".[dev,test,docs]"
   ```

3. **Verify installation**
   ```bash
   uv run pytest tests/unit/ -v --tb=short
   ```

## Development Process

### Branch Workflow

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes with clear, atomic commits

3. Push to your fork and submit a pull request

### Code Style

We follow these conventions:

- **Formatting**: Black (88 character line length)
- **Import sorting**: isort
- **Type hints**: Required for all public functions
- **Docstrings**: NumPy style

Run formatters before committing:
```bash
uv run black gridfia/ tests/
uv run isort gridfia/ tests/
```

### Type Checking

```bash
uv run mypy gridfia/
```

### Testing

- **Run all tests**: `uv run pytest`
- **Run with coverage**: `uv run pytest --cov=gridfia --cov-report=term-missing`
- **Run specific test file**: `uv run pytest tests/unit/test_api.py`

We maintain a minimum of 80% test coverage. All new features must include tests.

## Pull Request Process

1. **Update documentation** for any new features or API changes

2. **Add tests** for new functionality

3. **Ensure CI passes** - all tests and linting must pass

4. **Write a clear PR description** explaining:
   - What changes were made
   - Why they were made
   - How to test them

5. **Request review** from maintainers

### PR Title Convention

Use conventional commit format:
- `feat: Add new feature`
- `fix: Fix bug in X`
- `docs: Update documentation`
- `refactor: Refactor X for clarity`
- `test: Add tests for X`

## Project Structure

```
gridfia/
├── api.py              # Main API interface
├── config.py           # Configuration management
├── exceptions.py       # Custom exceptions
├── core/
│   ├── calculations/   # Forest metric calculations
│   ├── processors/     # Data processing pipelines
│   └── analysis/       # Statistical analysis
├── external/           # External service clients
├── utils/              # Utility modules
└── visualization/      # Mapping and plotting
```

## Adding New Calculations

1. Create a new class in `gridfia/core/calculations/`
2. Inherit from `ForestCalculation` base class
3. Implement `calculate()` and `validate_data()` methods
4. Register with `@registry.register("name")` decorator
5. Add tests in `tests/unit/test_calculations.py`

Example:
```python
from gridfia.core.calculations.base import ForestCalculation
from gridfia.core.calculations.registry import registry

@registry.register("my_calculation")
class MyCalculation(ForestCalculation):
    def __init__(self, **kwargs):
        super().__init__(
            name="my_calculation",
            description="Description of calculation",
            units="units",
            **kwargs
        )

    def calculate(self, biomass_data: np.ndarray, **kwargs) -> np.ndarray:
        # Implementation
        pass

    def validate_data(self, biomass_data: np.ndarray) -> bool:
        return biomass_data.ndim == 3 and biomass_data.shape[0] > 0
```

## Reporting Issues

When reporting issues, please include:

1. **Description** of the problem
2. **Steps to reproduce**
3. **Expected behavior**
4. **Actual behavior**
5. **Environment** (OS, Python version, GridFIA version)
6. **Error messages** or stack traces if applicable

## Questions?

- Open a [GitHub Discussion](https://github.com/mihiarc/gridfia/discussions)
- Check existing [Issues](https://github.com/mihiarc/gridfia/issues)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
