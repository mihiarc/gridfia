"""
Utilities for creating and managing Zarr stores for forest species data.

This module provides:
- ZarrStore: Unified class for reading GridFIA Zarr stores
- Functions for creating Zarr stores from GeoTIFFs
- Validation utilities for Zarr stores
"""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator
import numpy as np
import zarr
import zarr.storage
import zarr.codecs
import numcodecs
import rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS
import xarray as xr
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from ..exceptions import InvalidZarrStructure, SpeciesNotFound

console = Console()


class YearSlicedArray:
    """
    Wrapper that presents a 4D Zarr array as 3D by fixing the year index.

    The Icechunk BIGMAP datacube uses shape (year, species, y, x).
    This wrapper selects a single year index and exposes a 3D
    (species, y, x) interface compatible with ZarrStore expectations.

    Parameters
    ----------
    array : zarr.Array
        4D array with shape (year, species, y, x).
    year_index : int
        Index along the year dimension to select.
    """

    def __init__(self, array: zarr.Array, year_index: int = 0):
        self._array = array
        self._year_index = year_index

        if array.ndim != 4:
            raise ValueError(
                f"Expected 4D array (year, species, y, x), got ndim={array.ndim}"
            )
        if year_index < 0 or year_index >= array.shape[0]:
            raise IndexError(
                f"Year index {year_index} out of range for shape {array.shape}"
            )

    @property
    def shape(self) -> Tuple[int, int, int]:
        """3D shape: (species, height, width)."""
        return self._array.shape[1], self._array.shape[2], self._array.shape[3]

    @property
    def ndim(self) -> int:
        return 3

    @property
    def dtype(self) -> np.dtype:
        return self._array.dtype

    @property
    def chunks(self) -> Optional[Tuple[int, int, int]]:
        if hasattr(self._array, 'chunks') and self._array.chunks is not None:
            full_chunks = self._array.chunks
            # Drop the year chunk dimension
            return full_chunks[1], full_chunks[2], full_chunks[3]
        return None

    def __getitem__(self, key: Any) -> Any:
        """Index into the year-sliced view.

        Supports patterns used by ZarrStore consumers:
        - array[:]           -> full 3D load
        - array[i, :, :]     -> single species 2D layer
        - array[i]           -> single species 2D layer
        - array[:, y0:y1, x0:x1] -> spatial chunk across all species
        """
        if isinstance(key, slice) and key == slice(None):
            # array[:] -> load the full 3D slice for this year
            return self._array[self._year_index, :, :, :]

        if isinstance(key, (int, np.integer)):
            # array[i] -> single species layer
            return self._array[self._year_index, key, :, :]

        if isinstance(key, tuple):
            # Prepend year index to the key tuple
            full_key = (self._year_index,) + key
            return self._array[full_key]

        # Fallback: wrap the key
        return self._array[(self._year_index, key)]

    def __len__(self) -> int:
        return self.shape[0]

    def __repr__(self) -> str:
        return (
            f"YearSlicedArray(year_index={self._year_index}, "
            f"shape={self.shape}, dtype={self.dtype})"
        )


def create_blosc_codec(
    compression: str = 'zstd',
    compression_level: int = 3,
    shuffle: str = 'shuffle'
) -> zarr.codecs.BloscCodec:
    """
    Create a BloscCodec for Zarr v3 compression.

    This factory function provides a consistent way to create Blosc compression
    codecs across the codebase, handling the 'lz4' compression name mapping.

    Parameters
    ----------
    compression : str
        Compression algorithm name. Supported values:
        - 'lz4': LZ4 compression (fast, good compression ratio)
        - 'zstd': Zstandard compression (best compression ratio)
        - 'blosclz': BloscLZ compression
        - 'snappy': Snappy compression
    compression_level : int
        Compression level (0-9). Higher values = better compression but slower.
    shuffle : str
        Shuffle filter type. Options: 'shuffle', 'bitshuffle', 'noshuffle'

    Returns
    -------
    zarr.codecs.BloscCodec
        Configured Blosc codec for use with Zarr v3 arrays.

    Examples
    --------
    >>> codec = create_blosc_codec('zstd', compression_level=5)
    >>> array = root.create_array('data', shape=(100, 100), compressors=[codec])
    """
    return zarr.codecs.BloscCodec(cname=compression, clevel=compression_level, shuffle=shuffle)


class ZarrStore:
    """
    Unified interface for reading GridFIA Zarr stores.

    This class provides a standardized way to access Zarr stores created by GridFIA,
    handling both Zarr v2 and v3 formats transparently. It provides typed properties
    for common access patterns and includes context manager support for safe resource
    management.

    Attributes
    ----------
    path : Path
        Path to the Zarr store directory.
    biomass : zarr.Array
        The main biomass array with shape (species, height, width).
    species_codes : List[str]
        List of 4-digit FIA species codes.
    species_names : List[str]
        List of species common names.
    crs : CRS
        Coordinate reference system.
    transform : Affine
        Affine transformation for georeferencing.
    bounds : Tuple[float, float, float, float]
        Geographic bounds (left, bottom, right, top).
    num_species : int
        Number of species in the store.
    shape : Tuple[int, int, int]
        Shape of the biomass array (species, height, width).

    Examples
    --------
    Basic usage:

    >>> store = ZarrStore.from_path("data/montana.zarr")
    >>> print(f"Species: {store.num_species}")
    >>> print(f"Shape: {store.shape}")
    >>> data = store.biomass[0, :, :]  # Get first species layer

    Using context manager:

    >>> with ZarrStore.open("data/montana.zarr") as store:
    ...     richness = np.sum(store.biomass[:] > 0, axis=0)

    Iterating over species:

    >>> store = ZarrStore.from_path("data/forest.zarr")
    >>> for code, name in zip(store.species_codes, store.species_names):
    ...     print(f"{code}: {name}")
    """

    def __init__(
        self,
        root: zarr.Group,
        store: Optional[zarr.storage.LocalStore] = None,
        path: Optional[Path] = None
    ):
        """
        Initialize ZarrStore from an open Zarr group.

        Prefer using class methods `from_path()` or `open()` instead of
        calling this constructor directly.

        Parameters
        ----------
        root : zarr.Group
            Open Zarr group containing the biomass data.
        store : zarr.storage.LocalStore, optional
            The underlying storage object (for resource management).
        path : Path, optional
            Path to the Zarr store on disk.
        """
        self._root = root
        self._store = store
        self._path = path
        self._closed = False

        # Validate basic structure
        if 'biomass' not in self._root:
            raise InvalidZarrStructure(
                "Zarr store missing required 'biomass' array",
                zarr_path=str(path) if path else None,
                missing_attrs=['biomass']
            )

        self._biomass = self._root['biomass']

        # Cache commonly accessed values
        self._species_codes: Optional[List[str]] = None
        self._species_names: Optional[List[str]] = None
        self._crs: Optional[CRS] = None
        self._transform: Optional[Affine] = None
        self._bounds: Optional[Tuple[float, float, float, float]] = None

    @classmethod
    def from_path(cls, path: Union[str, Path], mode: str = 'r') -> 'ZarrStore':
        """
        Create a ZarrStore from a file path.

        This is the primary way to open an existing Zarr store for reading.

        Parameters
        ----------
        path : str or Path
            Path to the Zarr store directory.
        mode : str, default='r'
            Mode to open the store. Options:
            - 'r': Read-only (default)
            - 'r+': Read/write existing store

        Returns
        -------
        ZarrStore
            Initialized ZarrStore instance.

        Raises
        ------
        FileNotFoundError
            If the path does not exist.
        InvalidZarrStructure
            If the path is not a valid GridFIA Zarr store.

        Examples
        --------
        >>> store = ZarrStore.from_path("data/forest.zarr")
        >>> print(f"CRS: {store.crs}")
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Zarr store not found: {path}")

        try:
            # Try Zarr v3 API first (LocalStore)
            local_store = zarr.storage.LocalStore(path)
            root = zarr.open_group(store=local_store, mode=mode)
            return cls(root, store=local_store, path=path)
        except Exception as e:
            # Fall back to direct path opening (works for both v2 and v3)
            try:
                root = zarr.open_group(str(path), mode=mode)
                return cls(root, store=None, path=path)
            except Exception:
                raise InvalidZarrStructure(
                    f"Cannot open as Zarr group: {e}",
                    zarr_path=str(path)
                ) from e

    @classmethod
    def from_icechunk(
        cls,
        repo: Any,
        branch: str = "main",
        year: int = 2018,
    ) -> 'ZarrStore':
        """
        Create a ZarrStore from an Icechunk repository.

        Opens a readonly session on the given branch, selects the specified
        year from the 4D datacube, and presents a 3D (species, y, x) view
        compatible with all ZarrStore consumers.

        Parameters
        ----------
        repo : icechunk.Repository
            An Icechunk repository object. Can be obtained via
            ``icechunk.Repository.open()`` or ``arraylake.Client.get_repo()``.
        branch : str, default='main'
            Branch to read from.
        year : int, default=2018
            Year to select from the datacube's year dimension.

        Returns
        -------
        ZarrStore
            Initialized ZarrStore backed by Icechunk.

        Raises
        ------
        ImportError
            If the ``icechunk`` package is not installed.
        InvalidZarrStructure
            If the repository does not contain a valid BIGMAP datacube.
        ValueError
            If the requested year is not present in the datacube.

        Examples
        --------
        >>> import icechunk
        >>> storage = icechunk.s3_storage(bucket="my-bucket", prefix="gridfia/", from_env=True)
        >>> repo = icechunk.Repository.open(storage=storage)
        >>> store = ZarrStore.from_icechunk(repo, year=2018)
        >>> print(store.shape)
        (2, 84877, 164319)

        Using with Arraylake:

        >>> from arraylake import Client
        >>> client = Client()
        >>> repo = client.get_repo("ctrees/sfi_gridfia")
        >>> store = ZarrStore.from_icechunk(repo, branch="mihiar", year=2018)
        """
        try:
            import icechunk as _ic  # noqa: F401
        except ImportError:
            raise ImportError(
                "The 'icechunk' package is required for Icechunk backend support. "
                "Install it with: uv pip install gridfia[icechunk]"
            )

        # Open a readonly session on the specified branch
        session = repo.readonly_session(branch=branch)
        root = zarr.open_group(store=session.store, mode='r')

        if 'biomass' not in root:
            raise InvalidZarrStructure(
                "Icechunk repository missing required 'biomass' array",
                missing_attrs=['biomass']
            )

        biomass_array = root['biomass']

        if biomass_array.ndim != 4:
            raise InvalidZarrStructure(
                f"Expected 4D biomass array (year, species, y, x), "
                f"got ndim={biomass_array.ndim}",
                expected_shape=('year', 'species', 'y', 'x'),
                actual_shape=biomass_array.shape
            )

        # Resolve year index from metadata
        years_attr = list(root.attrs.get('years', []))
        if not years_attr:
            # If no years attribute, assume index 0 for any year
            year_index = 0
            console.print(
                f"[yellow]No 'years' attribute found, defaulting to year index 0"
            )
        elif year in years_attr:
            year_index = years_attr.index(year)
        else:
            raise ValueError(
                f"Year {year} not found in datacube. "
                f"Available years: {years_attr}"
            )

        # Wrap the 4D array to present a 3D view
        sliced_biomass = YearSlicedArray(biomass_array, year_index=year_index)

        # Build an adapter instance — we bypass __init__ validation since
        # we need to set _biomass to our wrapper instead of the raw array
        instance = cls.__new__(cls)
        instance._root = root
        instance._store = session.store
        instance._path = None
        instance._closed = False
        instance._biomass = sliced_biomass

        # Cache placeholders
        instance._species_codes = None
        instance._species_names = None
        instance._crs = None
        instance._transform = None
        instance._bounds = None

        # Eagerly resolve attribute mappings from Icechunk conventions
        # Icechunk datacube uses singular: species_code, species_name
        # GridFIA expects plural: species_codes, species_names
        if 'species_codes' in root:
            instance._species_codes = [str(c) for c in root['species_codes'][:] if c]
        elif 'species_code' in root.attrs:
            instance._species_codes = [str(c) for c in root.attrs['species_code']]
        elif 'species_code' in root:
            instance._species_codes = [str(c) for c in root['species_code'][:] if c]

        if 'species_names' in root:
            instance._species_names = [str(n) for n in root['species_names'][:] if n]
        elif 'species_name' in root.attrs:
            instance._species_names = [str(n) for n in root.attrs['species_name']]
        elif 'species_name' in root:
            instance._species_names = [str(n) for n in root['species_name'][:] if n]

        # Reconstruct transform from Icechunk components if needed
        if 'transform' in root.attrs:
            transform_list = root.attrs['transform']
            if len(transform_list) >= 6:
                instance._transform = Affine(*transform_list[:6])
        elif all(
            k in root.attrs
            for k in ('transform_origin_x', 'transform_origin_y', 'resolution_m')
        ):
            origin_x = root.attrs['transform_origin_x']
            origin_y = root.attrs['transform_origin_y']
            resolution = root.attrs['resolution_m']
            # Standard north-up raster: positive x resolution, negative y
            instance._transform = Affine(
                resolution, 0, origin_x,
                0, -resolution, origin_y
            )

        # Reconstruct CRS
        crs_str = root.attrs.get('crs', None)
        if crs_str:
            instance._crs = CRS.from_string(str(crs_str))

        # Reconstruct bounds from transform + shape if not stored directly
        if 'bounds' in root.attrs:
            bounds_list = root.attrs['bounds']
            if len(bounds_list) >= 4:
                instance._bounds = tuple(bounds_list[:4])
        elif instance._transform is not None:
            # Calculate bounds from transform and shape
            height_val = sliced_biomass.shape[1]
            width_val = sliced_biomass.shape[2]
            left = instance._transform.c
            right = left + width_val * instance._transform.a
            top = instance._transform.f
            bottom = top + height_val * instance._transform.e
            instance._bounds = (left, bottom, right, top)

        # Keep references alive
        instance._session = session
        instance._repo = repo
        instance._year = year
        instance._year_index = year_index

        return instance

    @classmethod
    @contextmanager
    def open(cls, path: Union[str, Path], mode: str = 'r') -> Iterator['ZarrStore']:
        """
        Context manager for safely opening and closing a ZarrStore.

        This ensures proper resource cleanup when done accessing the store.

        Parameters
        ----------
        path : str or Path
            Path to the Zarr store directory.
        mode : str, default='r'
            Mode to open the store ('r' for read-only, 'r+' for read/write).

        Yields
        ------
        ZarrStore
            Initialized ZarrStore instance.

        Examples
        --------
        >>> with ZarrStore.open("data/forest.zarr") as store:
        ...     total_biomass = np.sum(store.biomass[:])
        ...     print(f"Total biomass: {total_biomass:.2f}")
        """
        store = cls.from_path(path, mode=mode)
        try:
            yield store
        finally:
            store.close()

    @classmethod
    def is_valid_store(cls, path: Union[str, Path]) -> bool:
        """
        Check if a path contains a valid GridFIA Zarr store.

        This performs a quick validation without fully loading the store.

        Parameters
        ----------
        path : str or Path
            Path to check.

        Returns
        -------
        bool
            True if the path contains a valid GridFIA Zarr store.

        Examples
        --------
        >>> if ZarrStore.is_valid_store("data/forest.zarr"):
        ...     store = ZarrStore.from_path("data/forest.zarr")
        """
        path = Path(path)

        if not path.exists():
            return False

        try:
            # Try to open and check for required structure
            local_store = zarr.storage.LocalStore(path)
            root = zarr.open_group(store=local_store, mode='r')

            # Check for required arrays
            if 'biomass' not in root:
                return False

            # Check biomass is 3D (local) or 4D (Icechunk with year dimension)
            if root['biomass'].ndim not in (3, 4):
                return False

            return True

        except Exception:
            return False

    def close(self) -> None:
        """
        Close the Zarr store and release resources.

        After calling close(), the store should not be accessed.
        """
        self._closed = True
        # Clear cached values
        self._species_codes = None
        self._species_names = None
        self._crs = None
        self._transform = None
        self._bounds = None

    def _check_not_closed(self) -> None:
        """Raise an error if the store has been closed."""
        if self._closed:
            raise InvalidZarrStructure(
                "Cannot access closed ZarrStore",
                zarr_path=str(self._path) if self._path else None
            )

    @property
    def path(self) -> Optional[Path]:
        """Path to the Zarr store directory."""
        return self._path

    @property
    def biomass(self) -> zarr.Array:
        """
        The main biomass array.

        Shape is (species, height, width) where:
        - species: Number of species layers (index 0 is often total biomass)
        - height: Number of rows in the raster
        - width: Number of columns in the raster

        Values are typically in Mg/ha (megagrams per hectare).

        Returns
        -------
        zarr.Array
            3D array of biomass values.
        """
        self._check_not_closed()
        return self._biomass

    @property
    def species_codes(self) -> List[str]:
        """
        List of 4-digit FIA species codes.

        The first code ('0000') typically represents total biomass.
        Codes are zero-padded 4-digit strings (e.g., '0202' for Douglas-fir).

        Returns
        -------
        List[str]
            Species codes in order matching the biomass array's first dimension.
        """
        self._check_not_closed()

        if self._species_codes is None:
            if 'species_codes' in self._root:
                codes_array = self._root['species_codes'][:]
                self._species_codes = [str(c) for c in codes_array if c]
            else:
                # Fall back to attribute
                self._species_codes = list(self._root.attrs.get('species_codes', []))

        return self._species_codes

    @property
    def species_names(self) -> List[str]:
        """
        List of species common names.

        Names correspond to species codes and are in the same order as the
        biomass array's first dimension.

        Returns
        -------
        List[str]
            Species names (e.g., 'Douglas-fir', 'Ponderosa Pine').
        """
        self._check_not_closed()

        if self._species_names is None:
            if 'species_names' in self._root:
                names_array = self._root['species_names'][:]
                self._species_names = [str(n) for n in names_array if n]
            else:
                # Fall back to attribute
                self._species_names = list(self._root.attrs.get('species_names', []))

        return self._species_names

    @property
    def crs(self) -> CRS:
        """
        Coordinate reference system for the data.

        Returns
        -------
        rasterio.crs.CRS
            The CRS object (default is EPSG:3857 / Web Mercator if not specified).
        """
        self._check_not_closed()

        if self._crs is None:
            crs_str = self._root.attrs.get('crs', 'EPSG:3857')
            self._crs = CRS.from_string(crs_str)

        return self._crs

    @property
    def transform(self) -> Affine:
        """
        Affine transformation for georeferencing.

        The transform maps pixel coordinates to geographic coordinates:
        geo_x, geo_y = transform * (pixel_x, pixel_y)

        Returns
        -------
        rasterio.transform.Affine
            Affine transformation matrix.
        """
        self._check_not_closed()

        if self._transform is None:
            transform_list = self._root.attrs.get('transform', [1, 0, 0, 0, -1, 0])
            if len(transform_list) >= 6:
                self._transform = Affine(*transform_list[:6])
            else:
                self._transform = Affine.identity()

        return self._transform

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """
        Geographic bounds of the data.

        Returns
        -------
        Tuple[float, float, float, float]
            Bounds as (left, bottom, right, top) in the store's CRS.
        """
        self._check_not_closed()

        if self._bounds is None:
            bounds_list = self._root.attrs.get('bounds', [0, 0, 1, 1])
            if len(bounds_list) >= 4:
                self._bounds = tuple(bounds_list[:4])
            else:
                # Calculate from transform and shape
                height, width = self.shape[1], self.shape[2]
                left = self.transform.c
                right = left + width * self.transform.a
                top = self.transform.f
                bottom = top + height * self.transform.e
                self._bounds = (left, bottom, right, top)

        return self._bounds

    @property
    def num_species(self) -> int:
        """
        Number of species layers in the store.

        This includes the total biomass layer if present (typically at index 0).

        Returns
        -------
        int
            Count of species layers.
        """
        self._check_not_closed()
        return self._root.attrs.get('num_species', self._biomass.shape[0])

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        Shape of the biomass array.

        Returns
        -------
        Tuple[int, int, int]
            Shape as (species, height, width).
        """
        self._check_not_closed()
        return self._biomass.shape

    @property
    def height(self) -> int:
        """Number of rows in the raster."""
        self._check_not_closed()
        return self.shape[1]

    @property
    def width(self) -> int:
        """Number of columns in the raster."""
        self._check_not_closed()
        return self.shape[2]

    @property
    def chunks(self) -> Optional[Tuple[int, int, int]]:
        """
        Chunk size of the biomass array.

        Returns
        -------
        Tuple[int, int, int] or None
            Chunk dimensions or None if not chunked.
        """
        self._check_not_closed()
        return self._biomass.chunks if hasattr(self._biomass, 'chunks') else None

    @property
    def dtype(self) -> np.dtype:
        """Data type of the biomass array."""
        self._check_not_closed()
        return self._biomass.dtype

    @property
    def attrs(self) -> Dict[str, Any]:
        """
        All attributes from the root group.

        Returns
        -------
        Dict[str, Any]
            Dictionary of all stored attributes.
        """
        self._check_not_closed()
        return dict(self._root.attrs)

    def get_species_index(self, species_code: str) -> int:
        """
        Get the array index for a species code.

        Parameters
        ----------
        species_code : str
            4-digit FIA species code.

        Returns
        -------
        int
            Index in the biomass array's first dimension.

        Raises
        ------
        SpeciesNotFound
            If the species code is not in the store.

        Examples
        --------
        >>> store = ZarrStore.from_path("data/forest.zarr")
        >>> idx = store.get_species_index("0202")
        >>> douglas_fir = store.biomass[idx, :, :]
        """
        self._check_not_closed()

        try:
            return self.species_codes.index(species_code)
        except ValueError:
            raise SpeciesNotFound(
                f"Species code '{species_code}' not found in store",
                species_code=species_code,
                available_species=self.species_codes
            )

    def get_species_layer(self, species_code: str) -> np.ndarray:
        """
        Get the biomass layer for a specific species.

        Parameters
        ----------
        species_code : str
            4-digit FIA species code.

        Returns
        -------
        np.ndarray
            2D array of biomass values for the species.

        Raises
        ------
        SpeciesNotFound
            If the species code is not in the store.

        Examples
        --------
        >>> store = ZarrStore.from_path("data/forest.zarr")
        >>> ponderosa = store.get_species_layer("0122")
        >>> print(f"Max biomass: {ponderosa.max():.2f} Mg/ha")
        """
        idx = self.get_species_index(species_code)
        return self.biomass[idx, :, :]

    def get_species_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all species in the store.

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries with 'index', 'code', and 'name' keys.

        Examples
        --------
        >>> store = ZarrStore.from_path("data/forest.zarr")
        >>> for sp in store.get_species_info():
        ...     print(f"{sp['index']}: {sp['code']} - {sp['name']}")
        """
        self._check_not_closed()

        species_info = []
        codes = self.species_codes
        names = self.species_names

        for i in range(self.num_species):
            try:
                code = codes[i] if i < len(codes) else f"{i:04d}"
                name = names[i] if i < len(names) else f"Species {i}"
            except (IndexError, KeyError):
                code = f"{i:04d}"
                name = f"Species {i}"

            species_info.append({
                'index': i,
                'code': str(code),
                'name': str(name)
            })

        return species_info

    def get_extent(self) -> Tuple[float, float, float, float]:
        """
        Get extent for matplotlib plotting.

        Returns extent as (left, right, bottom, top) suitable for
        use with imshow's extent parameter.

        Returns
        -------
        Tuple[float, float, float, float]
            Extent as (left, right, bottom, top).
        """
        self._check_not_closed()

        height, width = self.shape[1], self.shape[2]
        transform = self.transform

        left = transform.c
        right = transform.c + width * transform.a
        top = transform.f
        bottom = transform.f + height * transform.e

        return (left, right, bottom, top)

    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of the Zarr store.

        Returns
        -------
        Dict[str, Any]
            Dictionary with store metadata and statistics.
        """
        self._check_not_closed()

        return {
            'path': str(self._path) if self._path else None,
            'shape': self.shape,
            'chunks': self.chunks,
            'dtype': str(self.dtype),
            'num_species': self.num_species,
            'crs': str(self.crs),
            'bounds': self.bounds,
            'transform': list(self.transform)[:6],
            'species_codes': self.species_codes,
            'species_names': self.species_names
        }

    def __repr__(self) -> str:
        if self._closed:
            return "ZarrStore(closed)"
        if hasattr(self, '_repo') and self._repo is not None:
            return (
                f"ZarrStore(icechunk, year={self._year}, shape={self.shape}, "
                f"species={self.num_species})"
            )
        return (
            f"ZarrStore(path={self._path}, shape={self.shape}, "
            f"species={self.num_species})"
        )

    def __enter__(self) -> 'ZarrStore':
        """Support for context manager protocol."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close store when exiting context."""
        self.close()


def create_expandable_zarr_from_base_raster(
    base_raster_path: Union[str, Path],
    zarr_path: Union[str, Path],
    max_species: int = 350,
    chunk_size: Tuple[int, int, int] = (1, 1000, 1000),
    compression: str = 'lz4',
    compression_level: int = 5
) -> zarr.Group:
    """
    Create an expandable Zarr store from a base raster file.
    
    Args:
        base_raster_path: Path to the base raster (e.g., total biomass or first species)
        zarr_path: Path where the Zarr store will be created
        max_species: Maximum number of species to allocate space for
        chunk_size: Chunk dimensions (species, height, width)
        compression: Compression algorithm to use
        compression_level: Compression level
        
    Returns:
        zarr.Group: The created Zarr group
    """
    console.print(f"[cyan]Creating Zarr store from base raster: {base_raster_path}")
    
    # Read base raster metadata
    with rasterio.open(base_raster_path) as src:
        height = src.height
        width = src.width
        crs = src.crs
        transform = src.transform
        bounds = src.bounds
        dtype = src.dtypes[0]
        
        # Read the data
        base_data = src.read(1)
    
    # Create Zarr store (Zarr v3 API)
    store = zarr.storage.LocalStore(zarr_path)
    root = zarr.open_group(store=store, mode='w')
    
    # Create the main data array with compression
    codec = create_blosc_codec(compression, compression_level)

    # Initialize with zeros
    data_array = root.create_array(
        'biomass',
        shape=(max_species, height, width),
        chunks=chunk_size,
        dtype=dtype,
        compressors=[codec],
        fill_value=0
    )
    
    # Add the base data as the first layer (index 0 for total biomass)
    data_array[0, :, :] = base_data
    
    # Store metadata
    root.attrs['crs'] = crs.to_string()
    root.attrs['transform'] = list(transform)
    root.attrs['bounds'] = list(bounds)
    root.attrs['width'] = width
    root.attrs['height'] = height
    root.attrs['num_species'] = 1  # Will be updated as species are added
    
    # Create species metadata arrays
    root.create_array(
        'species_codes',
        shape=(max_species,),
        dtype='<U10',
        fill_value=''
    )
    
    root.create_array(
        'species_names',
        shape=(max_species,),
        dtype='<U100',
        fill_value=''
    )
    
    # Set first entry as total biomass
    root['species_codes'][0] = '0000'
    root['species_names'][0] = 'Total Biomass'
    
    console.print(f"[green]✓ Created Zarr store with shape: {data_array.shape}")
    console.print(f"[green]✓ Chunk size: {chunk_size}")
    console.print(f"[green]✓ Compression: {compression} (level {compression_level})")
    
    return root


def append_species_to_zarr(
    zarr_path: Union[str, Path],
    species_raster_path: Union[str, Path],
    species_code: str,
    species_name: str,
    validate_alignment: bool = True
) -> int:
    """
    Append a species raster to an existing Zarr store.
    
    Args:
        zarr_path: Path to the existing Zarr store
        species_raster_path: Path to the species raster file
        species_code: Species code (e.g., '0202')
        species_name: Species common name (e.g., 'Douglas-fir')
        validate_alignment: Whether to validate spatial alignment
        
    Returns:
        int: The index where the species was added
    """
    console.print(f"[cyan]Adding species {species_code} - {species_name}")
    
    # Open Zarr store (Zarr v3 API)
    store = zarr.storage.LocalStore(zarr_path)
    root = zarr.open_group(store=store, mode='r+')
    
    # Get current number of species
    current_num = root.attrs['num_species']
    
    # Read species raster
    with rasterio.open(species_raster_path) as src:
        species_data = src.read(1)
        
        if validate_alignment:
            # Validate spatial alignment
            zarr_transform = Affine(*root.attrs['transform'])
            zarr_bounds = root.attrs['bounds']
            zarr_crs = CRS.from_string(root.attrs['crs'])
            
            if not np.allclose(src.transform, zarr_transform, rtol=1e-5):
                raise InvalidZarrStructure(
                    f"Transform mismatch for species {species_code}",
                    zarr_path=str(zarr_path)
                )

            if not np.allclose(src.bounds, zarr_bounds, rtol=1e-5):
                raise InvalidZarrStructure(
                    f"Bounds mismatch for species {species_code}",
                    zarr_path=str(zarr_path)
                )
            
            if src.crs != zarr_crs:
                console.print(f"[yellow]Warning: CRS mismatch. Expected {zarr_crs}, got {src.crs}")
    
    # Add species data
    root['biomass'][current_num, :, :] = species_data
    
    # Update metadata
    root['species_codes'][current_num] = species_code
    root['species_names'][current_num] = species_name
    root.attrs['num_species'] = current_num + 1
    
    console.print(f"[green]✓ Added {species_name} at index {current_num}")
    
    return current_num


def batch_append_species_from_dir(
    zarr_path: Union[str, Path],
    raster_dir: Union[str, Path],
    species_mapping: Dict[str, str],
    pattern: str = "*.tif",
    validate_alignment: bool = True
) -> None:
    """
    Batch append multiple species rasters from a directory.
    
    Args:
        zarr_path: Path to the existing Zarr store
        raster_dir: Directory containing species raster files
        species_mapping: Dictionary mapping species codes to names
        pattern: File pattern to match
        validate_alignment: Whether to validate spatial alignment
    """
    raster_dir = Path(raster_dir)
    raster_files = sorted(raster_dir.glob(pattern))
    
    if not raster_files:
        console.print(f"[red]No files found matching pattern {pattern} in {raster_dir}")
        return
    
    console.print(f"[cyan]Found {len(raster_files)} raster files to process")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Adding species to Zarr", total=len(raster_files))
        
        for raster_file in raster_files:
            # Extract species code from filename
            filename = raster_file.stem
            species_code = None
            
            # Try to find species code in filename
            for code in species_mapping:
                if code in filename:
                    species_code = code
                    break
            
            if species_code:
                species_name = species_mapping[species_code]
                try:
                    append_species_to_zarr(
                        zarr_path,
                        raster_file,
                        species_code,
                        species_name,
                        validate_alignment
                    )
                except Exception as e:
                    console.print(f"[red]Error adding {species_code}: {e}")
            else:
                console.print(f"[yellow]Warning: Could not find species code in {filename}")
            
            progress.update(task, advance=1)


def create_zarr_from_geotiffs(
    output_zarr_path: Union[str, Path],
    geotiff_paths: List[Union[str, Path]],
    species_codes: List[str],
    species_names: List[str],
    chunk_size: Tuple[int, int, int] = (1, 1000, 1000),
    compression: str = 'lz4',
    compression_level: int = 5,
    include_total: bool = True
) -> None:
    """
    Create a Zarr store from multiple GeoTIFF files.
    
    Args:
        output_zarr_path: Path for the output Zarr store
        geotiff_paths: List of paths to GeoTIFF files
        species_codes: List of species codes corresponding to each GeoTIFF
        species_names: List of species names corresponding to each GeoTIFF
        chunk_size: Chunk dimensions (species, height, width)
        compression: Compression algorithm
        compression_level: Compression level
        include_total: Whether to calculate and include total biomass as first layer
    """
    if len(geotiff_paths) != len(species_codes) or len(geotiff_paths) != len(species_names):
        raise InvalidZarrStructure(
            f"Number of paths ({len(geotiff_paths)}), codes ({len(species_codes)}), "
            f"and names ({len(species_names)}) must match",
            zarr_path=str(output_zarr_path)
        )
    
    console.print(f"[cyan]Creating Zarr store from {len(geotiff_paths)} GeoTIFF files")
    
    # Read first raster to get dimensions and metadata
    with rasterio.open(geotiff_paths[0]) as src:
        height = src.height
        width = src.width
        crs = src.crs
        transform = src.transform
        bounds = src.bounds
        dtype = src.dtypes[0]
    
    # Determine number of layers
    num_layers = len(geotiff_paths) + (1 if include_total else 0)
    
    # Create Zarr store (Zarr v3 API)
    store = zarr.storage.LocalStore(output_zarr_path)
    root = zarr.open_group(store=store, mode='w')
    
    # Create main data array with compression
    codec = create_blosc_codec(compression, compression_level)

    data_array = root.create_array(
        'biomass',
        shape=(num_layers, height, width),
        chunks=chunk_size,
        dtype=dtype,
        compressors=[codec],
        fill_value=0
    )
    
    # Create metadata arrays
    codes_array = root.create_array(
        'species_codes',
        shape=(num_layers,),
        dtype='<U10',
        fill_value=''
    )
    
    names_array = root.create_array(
        'species_names',
        shape=(num_layers,),
        dtype='<U100',
        fill_value=''
    )
    
    # Store spatial metadata
    root.attrs['crs'] = crs.to_string()
    root.attrs['transform'] = list(transform)
    root.attrs['bounds'] = list(bounds)
    root.attrs['width'] = width
    root.attrs['height'] = height
    
    # Process each species
    start_idx = 1 if include_total else 0
    total_biomass = np.zeros((height, width), dtype=dtype)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Processing species", total=len(geotiff_paths))
        
        for i, (path, code, name) in enumerate(zip(geotiff_paths, species_codes, species_names)):
            with rasterio.open(path) as src:
                data = src.read(1)
                
                # Validate alignment
                if src.height != height or src.width != width:
                    raise InvalidZarrStructure(
                        f"Dimension mismatch for {name}: expected ({height}, {width}), "
                        f"got ({src.height}, {src.width})",
                        zarr_path=str(output_zarr_path),
                        expected_shape=(None, height, width),
                        actual_shape=(None, src.height, src.width)
                    )
                if not np.allclose(src.transform, transform, rtol=1e-5):
                    raise InvalidZarrStructure(
                        f"Transform mismatch for {name}",
                        zarr_path=str(output_zarr_path)
                    )
                
                # Add to zarr
                idx = start_idx + i
                data_array[idx, :, :] = data
                codes_array[idx] = code
                names_array[idx] = name
                
                # Accumulate for total
                if include_total:
                    total_biomass += data
                
                progress.update(task, advance=1)
    
    # Add total biomass if requested
    if include_total:
        data_array[0, :, :] = total_biomass
        codes_array[0] = '0000'
        names_array[0] = 'Total Biomass'
    
    root.attrs['num_species'] = num_layers
    
    console.print(f"[green]✓ Created Zarr store at {output_zarr_path}")
    console.print(f"[green]✓ Shape: {data_array.shape}")
    console.print(f"[green]✓ Species: {', '.join(species_names)}")


def validate_zarr_store(zarr_path: Union[str, Path]) -> Dict:
    """
    Validate and summarize a Zarr store.
    
    Args:
        zarr_path: Path to the Zarr store
        
    Returns:
        Dict: Summary information about the Zarr store
    """
    store = zarr.storage.LocalStore(zarr_path)
    root = zarr.open_group(store=store, mode='r')
    
    info = {
        'path': str(zarr_path),
        'shape': root['biomass'].shape,
        'chunks': root['biomass'].chunks,
        'dtype': str(root['biomass'].dtype),
        'compression': 'blosc' if hasattr(root['biomass'], 'codecs') else None,
        'num_species': root.attrs.get('num_species', 0),
        'crs': root.attrs.get('crs'),
        'bounds': root.attrs.get('bounds'),
        'species': []
    }
    
    # Get species information
    if 'species_codes' in root:
        codes_arr = root['species_codes'][:]
        names_arr = root['species_names'][:] if 'species_names' in root else []

        for i in range(info['num_species']):
            if i < len(codes_arr):
                code = codes_arr[i]
                name = names_arr[i] if i < len(names_arr) else ''
                if code:  # Skip empty entries
                    info['species'].append({
                        'index': i,
                        'code': str(code),
                        'name': str(name) if name else f"Species {code}"
                    })
    
    return info