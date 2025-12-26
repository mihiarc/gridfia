"""
Domain-specific exceptions for GridFIA.

This module defines a hierarchy of exceptions used throughout the GridFIA package
to provide clear, actionable error messages for common failure scenarios.

Exception Hierarchy:
    GridFIAException (base)
    |-- InvalidZarrStructure
    |-- SpeciesNotFound
    |-- CalculationFailed
    |-- APIConnectionError
    |-- InvalidLocationConfig
    |-- DownloadError
    |-- CircuitBreakerOpen
"""

from typing import Optional, List, Any


class GridFIAException(Exception):
    """
    Base exception for all GridFIA errors.

    All GridFIA-specific exceptions inherit from this class, making it easy
    to catch any GridFIA error with a single except clause.

    Examples
    --------
    >>> try:
    ...     api.calculate_metrics("invalid.zarr")
    ... except GridFIAException as e:
    ...     print(f"GridFIA error: {e}")
    """

    def __init__(self, message: str, details: Optional[dict] = None):
        """
        Initialize the exception.

        Parameters
        ----------
        message : str
            Human-readable error message
        details : dict, optional
            Additional context about the error
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


class InvalidZarrStructure(GridFIAException):
    """
    Raised when Zarr array has invalid structure or missing attributes.

    This exception is raised when:
    - The Zarr store cannot be opened
    - Required arrays (biomass, species_codes) are missing
    - Array dimensions are incorrect
    - Required metadata attributes are missing

    Examples
    --------
    >>> try:
    ...     api.calculate_metrics("corrupt.zarr")
    ... except InvalidZarrStructure as e:
    ...     print(f"Invalid Zarr: {e}")
    """

    def __init__(
        self,
        message: str,
        zarr_path: Optional[str] = None,
        missing_attrs: Optional[List[str]] = None,
        expected_shape: Optional[tuple] = None,
        actual_shape: Optional[tuple] = None
    ):
        details = {}
        if zarr_path:
            details["path"] = zarr_path
        if missing_attrs:
            details["missing_attrs"] = missing_attrs
        if expected_shape:
            details["expected_shape"] = expected_shape
        if actual_shape:
            details["actual_shape"] = actual_shape

        super().__init__(message, details)
        self.zarr_path = zarr_path
        self.missing_attrs = missing_attrs
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape


class SpeciesNotFound(GridFIAException):
    """
    Raised when requested species code is not found.

    This exception is raised when:
    - A species code does not exist in the FIA BIGMAP service
    - A species code is not present in a Zarr store
    - No raster function matches the requested species

    Examples
    --------
    >>> try:
    ...     api.download_species(state="MT", species_codes=["9999"])
    ... except SpeciesNotFound as e:
    ...     print(f"Species not found: {e}")
    """

    def __init__(
        self,
        message: str,
        species_code: Optional[str] = None,
        available_species: Optional[List[str]] = None
    ):
        details = {}
        if species_code:
            details["species_code"] = species_code
        if available_species:
            details["available_count"] = len(available_species)

        super().__init__(message, details)
        self.species_code = species_code
        self.available_species = available_species


class CalculationFailed(GridFIAException):
    """
    Raised when a calculation fails to complete.

    This exception is raised when:
    - A registered calculation encounters an error
    - Input data validation fails for a calculation
    - The calculation registry cannot find the requested calculation

    Examples
    --------
    >>> try:
    ...     api.calculate_metrics("data.zarr", calculations=["invalid_calc"])
    ... except CalculationFailed as e:
    ...     print(f"Calculation error: {e}")
    """

    def __init__(
        self,
        message: str,
        calculation_name: Optional[str] = None,
        available_calculations: Optional[List[str]] = None,
        original_error: Optional[Exception] = None
    ):
        details = {}
        if calculation_name:
            details["calculation"] = calculation_name
        if available_calculations:
            details["available"] = available_calculations
        if original_error:
            details["original_error"] = str(original_error)

        super().__init__(message, details)
        self.calculation_name = calculation_name
        self.available_calculations = available_calculations
        self.original_error = original_error


class APIConnectionError(GridFIAException):
    """
    Raised when connection to FIA API fails.

    This exception is raised when:
    - Network connection to FIA BIGMAP service fails
    - Request timeout occurs
    - Server returns error status codes (after retries)
    - Rate limiting prevents successful request

    Examples
    --------
    >>> try:
    ...     api.list_species()
    ... except APIConnectionError as e:
    ...     print(f"API error: {e}")
    """

    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        original_error: Optional[Exception] = None
    ):
        details = {}
        if url:
            details["url"] = url
        if status_code:
            details["status_code"] = status_code
        if original_error:
            details["original_error"] = str(original_error)

        super().__init__(message, details)
        self.url = url
        self.status_code = status_code
        self.original_error = original_error


class InvalidLocationConfig(GridFIAException):
    """
    Raised when location configuration is invalid.

    This exception is raised when:
    - State name or abbreviation is not recognized
    - County is not found within the specified state
    - Bounding box coordinates are invalid
    - Configuration file is malformed or missing required fields

    Examples
    --------
    >>> try:
    ...     api.get_location_config(state="InvalidState")
    ... except InvalidLocationConfig as e:
    ...     print(f"Location error: {e}")
    """

    def __init__(
        self,
        message: str,
        location_type: Optional[str] = None,
        location_name: Optional[str] = None,
        state: Optional[str] = None,
        county: Optional[str] = None
    ):
        details = {}
        if location_type:
            details["type"] = location_type
        if location_name:
            details["name"] = location_name
        if state:
            details["state"] = state
        if county:
            details["county"] = county

        super().__init__(message, details)
        self.location_type = location_type
        self.location_name = location_name
        self.state = state
        self.county = county


class DownloadError(GridFIAException):
    """
    Raised when data download fails.

    This exception is raised when:
    - Raster export from FIA service fails
    - Downloaded file is empty or corrupt
    - No files were successfully downloaded in a batch operation
    - Output directory cannot be created or written to

    Examples
    --------
    >>> try:
    ...     api.download_species(state="MT", species_codes=["0202"])
    ... except DownloadError as e:
    ...     print(f"Download error: {e}")
    """

    def __init__(
        self,
        message: str,
        species_code: Optional[str] = None,
        output_path: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        details = {}
        if species_code:
            details["species_code"] = species_code
        if output_path:
            details["output_path"] = output_path
        if original_error:
            details["original_error"] = str(original_error)

        super().__init__(message, details)
        self.species_code = species_code
        self.output_path = output_path
        self.original_error = original_error


class CircuitBreakerOpen(GridFIAException):
    """
    Raised when circuit breaker is in OPEN state and requests are blocked.

    This exception is raised when:
    - Too many consecutive failures have occurred
    - The circuit breaker is protecting against cascading failures
    - The recovery timeout has not yet elapsed

    The circuit breaker will automatically transition to HALF_OPEN state
    after the recovery timeout to test if the service has recovered.

    Examples
    --------
    >>> try:
    ...     api.list_species()
    ... except CircuitBreakerOpen as e:
    ...     print(f"Service unavailable: {e}")
    ...     print(f"Retry after: {e.retry_after} seconds")
    """

    def __init__(
        self,
        message: str,
        failure_count: Optional[int] = None,
        failure_threshold: Optional[int] = None,
        retry_after: Optional[float] = None,
        last_failure_time: Optional[float] = None
    ):
        details = {}
        if failure_count is not None:
            details["failure_count"] = failure_count
        if failure_threshold is not None:
            details["failure_threshold"] = failure_threshold
        if retry_after is not None:
            details["retry_after_seconds"] = round(retry_after, 2)
        if last_failure_time is not None:
            details["last_failure_time"] = last_failure_time

        super().__init__(message, details)
        self.failure_count = failure_count
        self.failure_threshold = failure_threshold
        self.retry_after = retry_after
        self.last_failure_time = last_failure_time
