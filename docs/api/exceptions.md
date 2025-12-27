# Exceptions

GridFIA provides a hierarchy of domain-specific exceptions for clear, actionable error handling.
All exceptions inherit from `GridFIAException`, allowing you to catch any GridFIA error
with a single except clause.

## Exception Hierarchy

```
GridFIAException (base)
├── InvalidZarrStructure    # Zarr store validation errors
├── SpeciesNotFound         # Species code resolution errors
├── CalculationFailed       # Calculation execution errors
├── APIConnectionError      # Network/API errors
├── InvalidLocationConfig   # Geographic configuration errors
├── DownloadError          # Data download failures
└── CircuitBreakerOpen     # Service protection (circuit breaker)
```

## Quick Reference

| Exception | Raised When |
|-----------|-------------|
| `GridFIAException` | Base class for all GridFIA errors |
| `InvalidZarrStructure` | Zarr store is corrupt or missing required data |
| `SpeciesNotFound` | Requested species code doesn't exist |
| `CalculationFailed` | Calculation encounters an error |
| `APIConnectionError` | FIA BIGMAP service connection fails |
| `InvalidLocationConfig` | State/county/bbox configuration is invalid |
| `DownloadError` | Raster download fails |
| `CircuitBreakerOpen` | Too many consecutive failures |

## Usage Examples

### Catching All GridFIA Errors

```python
from gridfia import GridFIA, GridFIAException

api = GridFIA()

try:
    results = api.calculate_metrics("data.zarr")
except GridFIAException as e:
    print(f"GridFIA error: {e.message}")
    print(f"Details: {e.details}")
```

### Specific Error Handling

```python
from gridfia import GridFIA
from gridfia.exceptions import (
    InvalidZarrStructure,
    SpeciesNotFound,
    CalculationFailed,
    APIConnectionError,
    InvalidLocationConfig,
    DownloadError,
    CircuitBreakerOpen,
)

api = GridFIA()

try:
    # Attempt download
    files = api.download_species(
        state="Montana",
        species_codes=["0202"]
    )

    # Create Zarr
    zarr_path = api.create_zarr("downloads/", "data.zarr")

    # Calculate metrics
    results = api.calculate_metrics(zarr_path)

except InvalidLocationConfig as e:
    print(f"Location error: {e.message}")
    print(f"  State: {e.state}")
    print(f"  County: {e.county}")

except SpeciesNotFound as e:
    print(f"Species not found: {e.species_code}")
    if e.available_species:
        print(f"  Available: {e.available_species[:5]}...")

except DownloadError as e:
    print(f"Download failed: {e.message}")
    print(f"  Species: {e.species_code}")
    print(f"  Output: {e.output_path}")

except APIConnectionError as e:
    print(f"API error: {e.message}")
    print(f"  URL: {e.url}")
    print(f"  Status: {e.status_code}")

except CircuitBreakerOpen as e:
    print(f"Service unavailable: {e.message}")
    print(f"  Retry after: {e.retry_after} seconds")

except InvalidZarrStructure as e:
    print(f"Invalid Zarr: {e.message}")
    print(f"  Path: {e.zarr_path}")
    print(f"  Missing: {e.missing_attrs}")

except CalculationFailed as e:
    print(f"Calculation error: {e.message}")
    print(f"  Calculation: {e.calculation_name}")
    print(f"  Available: {e.available_calculations}")
```

## Exception Reference

### GridFIAException

::: gridfia.exceptions.GridFIAException
    options:
      show_root_heading: false
      heading_level: 4

### InvalidZarrStructure

::: gridfia.exceptions.InvalidZarrStructure
    options:
      show_root_heading: false
      heading_level: 4

### SpeciesNotFound

::: gridfia.exceptions.SpeciesNotFound
    options:
      show_root_heading: false
      heading_level: 4

### CalculationFailed

::: gridfia.exceptions.CalculationFailed
    options:
      show_root_heading: false
      heading_level: 4

### APIConnectionError

::: gridfia.exceptions.APIConnectionError
    options:
      show_root_heading: false
      heading_level: 4

### InvalidLocationConfig

::: gridfia.exceptions.InvalidLocationConfig
    options:
      show_root_heading: false
      heading_level: 4

### DownloadError

::: gridfia.exceptions.DownloadError
    options:
      show_root_heading: false
      heading_level: 4

### CircuitBreakerOpen

::: gridfia.exceptions.CircuitBreakerOpen
    options:
      show_root_heading: false
      heading_level: 4

## Error Recovery Patterns

### Retry with Exponential Backoff

```python
import time
from gridfia import GridFIA
from gridfia.exceptions import APIConnectionError, CircuitBreakerOpen

def download_with_retry(api, state, max_retries=3):
    """Download with exponential backoff on failure."""
    for attempt in range(max_retries):
        try:
            return api.download_species(state=state)

        except CircuitBreakerOpen as e:
            # Wait for circuit breaker to reset
            wait_time = e.retry_after or 60
            print(f"Circuit breaker open, waiting {wait_time}s...")
            time.sleep(wait_time)

        except APIConnectionError as e:
            # Exponential backoff
            wait_time = 2 ** attempt
            print(f"API error (attempt {attempt + 1}), retrying in {wait_time}s...")
            time.sleep(wait_time)

    raise RuntimeError(f"Failed after {max_retries} attempts")
```

### Graceful Degradation

```python
from gridfia import GridFIA
from gridfia.exceptions import CalculationFailed

api = GridFIA()

def calculate_all_metrics(zarr_path, calculations):
    """Calculate metrics with graceful degradation."""
    results = []
    failed = []

    for calc in calculations:
        try:
            result = api.calculate_metrics(
                zarr_path,
                calculations=[calc]
            )
            results.extend(result)
        except CalculationFailed as e:
            print(f"Warning: {calc} failed - {e.message}")
            failed.append(calc)

    if failed:
        print(f"Some calculations failed: {failed}")

    return results
```

### Validation Before Processing

```python
from pathlib import Path
from gridfia import GridFIA
from gridfia.exceptions import InvalidZarrStructure, SpeciesNotFound

api = GridFIA()

def validate_and_process(zarr_path, species_codes):
    """Validate inputs before expensive processing."""

    # Validate Zarr structure
    try:
        info = api.validate_zarr(zarr_path)
        print(f"Valid Zarr: {info['shape']}")
    except InvalidZarrStructure as e:
        return {"error": "invalid_zarr", "details": e.details}

    # Validate species codes exist in store
    available = info.get("species_codes", [])
    missing = [s for s in species_codes if s not in available]
    if missing:
        return {
            "error": "missing_species",
            "missing": missing,
            "available": available
        }

    # Proceed with processing
    return api.calculate_metrics(zarr_path)
```

## Logging Errors

All exceptions include structured details for logging:

```python
import logging
from gridfia import GridFIA, GridFIAException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api = GridFIA()

try:
    api.download_species(state="InvalidState")
except GridFIAException as e:
    logger.error(
        "GridFIA operation failed",
        extra={
            "error_type": type(e).__name__,
            "message": e.message,
            "details": e.details
        }
    )
```

## See Also

- [GridFIA Class](gridfia.md) - Main API methods
- [Configuration](config.md) - Settings that affect error behavior
