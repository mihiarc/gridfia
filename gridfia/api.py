"""
GridFIA API - Spatial raster analysis for USDA Forest Service BIGMAP data.

Part of the FIA Python Ecosystem:
- PyFIA: Survey/plot data analysis
- GridFIA: Spatial raster analysis (this package)
- PyFVS: Growth/yield simulation
- AskFIA: AI conversational interface

This module provides the primary API for GridFIA functionality.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import threading

import numpy as np
import xarray as xr
from pydantic import BaseModel, Field
from pyproj import Transformer

from .config import GridFIASettings, CalculationConfig, load_settings
from .core.processors.forest_metrics import ForestMetricsProcessor
from .external.fia_client import BigMapRestClient
from .utils.location_config import LocationConfig
from .utils.zarr_utils import create_zarr_from_geotiffs, validate_zarr_store, ZarrStore
from .visualization.mapper import ZarrMapper
from .core.calculations import registry
from .exceptions import (
    InvalidZarrStructure,
    SpeciesNotFound,
    CalculationFailed,
    InvalidLocationConfig,
    DownloadError,
)

logger = logging.getLogger(__name__)


class CalculationResult(BaseModel):
    """Result from a calculation operation."""
    name: str
    output_path: Path
    statistics: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SpeciesInfo(BaseModel):
    """Information about a tree species."""
    species_code: str = Field(pattern=r'^\d{4}$', description="4-digit FIA species code")
    common_name: str = Field(min_length=1, description="Common name of the species")
    scientific_name: str = Field(min_length=1, description="Scientific name of the species")
    function_name: Optional[str] = None


class GridFIA:
    """
    Main API interface for GridFIA spatial forest analysis.

    GridFIA provides spatial raster analysis of USDA Forest Service BIGMAP data,
    including species biomass mapping, diversity metrics, and visualization.

    Part of the FIA Python Ecosystem - use with PyFIA for survey data,
    PyFVS for growth simulation, and AskFIA for AI-powered queries.

    Examples
    --------
    >>> from gridfia import GridFIA
    >>> api = GridFIA()
    >>>
    >>> # Download species data for North Carolina
    >>> api.download_species(state="NC", species_codes=["0131", "0068"])
    >>>
    >>> # Create zarr store from downloaded data
    >>> api.create_zarr("downloads/", "data/nc_forest.zarr")
    >>>
    >>> # Calculate forest metrics
    >>> results = api.calculate_metrics(
    ...     "data/nc_forest.zarr",
    ...     calculations=["species_richness", "shannon_diversity"]
    ... )
    >>>
    >>> # Create visualization
    >>> api.create_maps("data/nc_forest.zarr", map_type="diversity")
    """

    def __init__(self, config: Optional[Union[str, Path, GridFIASettings]] = None):
        """
        Initialize GridFIA API.

        Parameters
        ----------
        config : str, Path, or GridFIASettings, optional
            Configuration file path or settings object.
            If None, uses default settings.
        """
        if config is None:
            self.settings = GridFIASettings()
        elif isinstance(config, (str, Path)):
            self.settings = load_settings(Path(config))
        else:
            self.settings = config

        # Lock for thread-safe lazy initialization of components
        self._init_lock = threading.Lock()
        self._rest_client = None
        self._processor = None
        
    @property
    def rest_client(self) -> BigMapRestClient:
        """Lazy-load REST client for FIA BIGMAP service (thread-safe)."""
        # Double-checked locking pattern for thread-safe lazy initialization
        if self._rest_client is None:
            with self._init_lock:
                # Check again after acquiring lock (another thread may have initialized)
                if self._rest_client is None:
                    self._rest_client = BigMapRestClient()
        return self._rest_client
    
    @property
    def processor(self) -> ForestMetricsProcessor:
        """Lazy-load forest metrics processor (thread-safe)."""
        # Double-checked locking pattern for thread-safe lazy initialization
        if self._processor is None:
            with self._init_lock:
                # Check again after acquiring lock (another thread may have initialized)
                if self._processor is None:
                    self._processor = ForestMetricsProcessor(self.settings)
        return self._processor
    
    def list_species(self) -> List[SpeciesInfo]:
        """
        List all available tree species from FIA BIGMAP service.
        
        Returns
        -------
        List[SpeciesInfo]
            List of available species with codes and names.
            
        Examples
        --------
        >>> api = GridFIA()
        >>> species = api.list_species()
        >>> print(f"Found {len(species)} species")
        >>> for s in species[:5]:
        ...     print(f"{s.species_code}: {s.common_name}")
        """
        species_data = self.rest_client.list_available_species()
        return [
            SpeciesInfo(
                species_code=s['species_code'],
                common_name=s['common_name'],
                scientific_name=s['scientific_name'],
                function_name=s.get('function_name')
            )
            for s in species_data
        ]
    
    def download_species(
        self,
        output_dir: Union[str, Path] = "downloads",
        species_codes: Optional[List[str]] = None,
        state: Optional[str] = None,
        county: Optional[str] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        location_config: Optional[Union[str, Path]] = None,
        crs: str = "102100"
    ) -> List[Path]:
        """
        Download species data from FIA BIGMAP service.
        
        Parameters
        ----------
        output_dir : str or Path, default="downloads"
            Directory to save downloaded files.
        species_codes : List[str], optional
            Specific species codes to download. If None, downloads all.
        state : str, optional
            State name or abbreviation.
        county : str, optional
            County name (requires state).
        bbox : Tuple[float, float, float, float], optional
            Custom bounding box (xmin, ymin, xmax, ymax).
        location_config : str or Path, optional
            Path to location configuration file.
        crs : str, default="102100"
            Coordinate reference system for bbox.
            
        Returns
        -------
        List[Path]
            Paths to downloaded files.
            
        Examples
        --------
        >>> api = GridFIA()
        >>> # Download for entire state
        >>> files = api.download_species(state="Montana", species_codes=["0202"])
        >>> 
        >>> # Download for specific county
        >>> files = api.download_species(
        ...     state="Texas", 
        ...     county="Harris",
        ...     species_codes=["0131", "0068"]
        ... )
        >>> 
        >>> # Download with custom bbox
        >>> files = api.download_species(
        ...     bbox=(-104, 44, -104.5, 44.5),
        ...     crs="4326"
        ... )
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine location and bbox
        location_name = "location"
        location_bbox = None
        bbox_crs = crs
        
        if location_config:
            config = LocationConfig(Path(location_config))
            location_name = config.location_name.lower().replace(' ', '_')
            location_bbox = config.web_mercator_bbox
            logger.info(f"Using location config: {config.location_name}")
            
        elif state:
            if county:
                config = LocationConfig.from_county(county, state)
                location_name = f"{county}_{state}".lower().replace(' ', '_')
            else:
                config = LocationConfig.from_state(state)
                location_name = state.lower().replace(' ', '_')
            
            location_bbox = config.web_mercator_bbox
            logger.info(f"Using {config.location_name} boundaries")
            
        elif bbox:
            # Transform bbox to Web Mercator if a different CRS is specified
            crs_normalized = str(crs).upper().replace("EPSG:", "")
            if crs_normalized not in ("102100", "3857"):
                # Need to transform from input CRS to Web Mercator
                try:
                    source_crs = f"EPSG:{crs_normalized}" if crs_normalized.isdigit() else crs
                    transformer = Transformer.from_crs(source_crs, "EPSG:3857", always_xy=True)
                    xmin, ymin = transformer.transform(bbox[0], bbox[1])
                    xmax, ymax = transformer.transform(bbox[2], bbox[3])
                    location_bbox = (xmin, ymin, xmax, ymax)
                    logger.info(f"Transformed bbox from {crs} to Web Mercator: {location_bbox}")
                except Exception as e:
                    raise InvalidLocationConfig(
                        f"Failed to transform bbox from CRS {crs} to Web Mercator: {e}",
                        location_type="bbox"
                    )
                # Always use Web Mercator for API calls
                bbox_crs = "102100"
            else:
                location_bbox = bbox

        else:
            raise InvalidLocationConfig(
                "Must specify state, bbox, or location_config",
                location_type="unknown"
            )

        if not location_bbox:
            raise InvalidLocationConfig(
                "Could not determine bounding box for location",
                location_name=location_name
            )
        
        # Download species data
        exported_files = self.rest_client.batch_export_location_species(
            bbox=location_bbox,
            output_dir=output_dir,
            species_codes=species_codes,
            location_name=location_name,
            bbox_srs=bbox_crs
        )
        
        logger.info(f"Downloaded {len(exported_files)} species rasters")
        return exported_files
    
    def create_zarr(
        self,
        input_dir: Union[str, Path],
        output_path: Union[str, Path],
        species_codes: Optional[List[str]] = None,
        chunk_size: Tuple[int, int, int] = (1, 1000, 1000),
        compression: str = "lz4",
        compression_level: int = 5,
        include_total: bool = True
    ) -> Path:
        """
        Create a Zarr store from GeoTIFF files.
        
        Parameters
        ----------
        input_dir : str or Path
            Directory containing GeoTIFF files.
        output_path : str or Path
            Output path for Zarr store.
        species_codes : List[str], optional
            Specific species codes to include.
        chunk_size : Tuple[int, int, int], default=(1, 1000, 1000)
            Chunk dimensions (species, height, width).
        compression : str, default="lz4"
            Compression algorithm.
        compression_level : int, default=5
            Compression level (1-9).
        include_total : bool, default=True
            Whether to include or calculate total biomass.
            
        Returns
        -------
        Path
            Path to created Zarr store.
            
        Examples
        --------
        >>> api = GridFIA()
        >>> zarr_path = api.create_zarr(
        ...     "downloads/montana_species/",
        ...     "data/montana.zarr",
        ...     chunk_size=(1, 2000, 2000)
        ... )
        >>> print(f"Created Zarr store at {zarr_path}")
        """
        input_dir = Path(input_dir)
        output_path = Path(output_path)
        
        if not input_dir.exists():
            raise DownloadError(
                f"Input directory does not exist: {input_dir}",
                output_path=str(input_dir)
            )
        
        # Find GeoTIFF files
        tiff_files = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))
        
        if not tiff_files:
            raise DownloadError(
                f"No GeoTIFF files found in {input_dir}",
                output_path=str(input_dir)
            )
        
        logger.info(f"Found {len(tiff_files)} GeoTIFF files")
        
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
                raise SpeciesNotFound(
                    f"No files found for species codes: {species_codes}",
                    available_species=species_codes
                )
        
        # Sort files for consistent ordering
        tiff_files.sort()
        
        # Extract species information from filenames
        import re
        file_species_codes = []
        file_species_names = []
        
        for f in tiff_files:
            filename = f.stem
            code = None
            name = filename
            
            # Look for 4-digit species code
            match = re.search(r'(\d{4})', filename)
            if match:
                code = match.group(1)
                # Try to extract name after code
                parts = filename.split(code)
                if len(parts) > 1:
                    name = parts[1].strip('_- ').replace('_', ' ')
            
            file_species_codes.append(code or filename[:4])
            file_species_names.append(name.title())
        
        # Create the Zarr store
        create_zarr_from_geotiffs(
            output_zarr_path=output_path,
            geotiff_paths=tiff_files,
            species_codes=file_species_codes,
            species_names=file_species_names,
            chunk_size=chunk_size,
            compression=compression,
            compression_level=compression_level,
            include_total=include_total
        )
        
        # Validate the created store
        info = validate_zarr_store(output_path)
        logger.info(f"Created Zarr store: shape={info['shape']}, species={info['num_species']}")
        
        return output_path
    
    def calculate_metrics(
        self,
        zarr_path: Union[str, Path],
        calculations: Optional[List[str]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        config: Optional[Union[str, Path, GridFIASettings]] = None
    ) -> List[CalculationResult]:
        """
        Calculate forest metrics from Zarr data.
        
        Parameters
        ----------
        zarr_path : str or Path
            Path to Zarr store containing biomass data.
        calculations : List[str], optional
            Specific calculations to run. If None, uses config or defaults.
        output_dir : str or Path, optional
            Output directory for results.
        config : str, Path, or GridFIASettings, optional
            Configuration to use for calculations.
            
        Returns
        -------
        List[CalculationResult]
            Results from each calculation.
            
        Examples
        --------
        >>> api = GridFIA()
        >>> results = api.calculate_metrics(
        ...     "data/forest.zarr",
        ...     calculations=["species_richness", "shannon_diversity", "total_biomass"]
        ... )
        >>> for r in results:
        ...     print(f"{r.name}: {r.output_path}")
        ...     print(f"  Stats: {r.statistics}")
        """
        zarr_path = Path(zarr_path)
        
        if not zarr_path.exists():
            raise InvalidZarrStructure(
                f"Zarr store not found: {zarr_path}",
                zarr_path=str(zarr_path)
            )

        # Load configuration if provided
        if config:
            if isinstance(config, (str, Path)):
                settings = load_settings(Path(config))
            else:
                settings = config
        else:
            settings = self.settings
        
        # Override output directory if specified
        if output_dir:
            settings.output_dir = Path(output_dir)
        
        # Override calculations if specified
        if calculations:
            # Validate calculations exist
            all_registered = registry.list_calculations()
            invalid_calcs = [c for c in calculations if c not in all_registered]
            if invalid_calcs:
                raise CalculationFailed(
                    f"Unknown calculations: {invalid_calcs}",
                    calculation_name=invalid_calcs[0] if len(invalid_calcs) == 1 else str(invalid_calcs),
                    available_calculations=all_registered
                )
            
            # Create calculation configs
            settings.calculations = [
                CalculationConfig(name=calc_name, enabled=True)
                for calc_name in calculations
            ]
        
        # Run calculations
        processor = ForestMetricsProcessor(settings)
        output_paths = processor.run_calculations(str(zarr_path))
        
        # Convert to results
        results = []
        for name, path in output_paths.items():
            results.append(
                CalculationResult(
                    name=name,
                    output_path=Path(path),
                    statistics={},  # Could be enhanced to include actual stats
                    metadata={"zarr_path": str(zarr_path)}
                )
            )
        
        return results
    
    def create_maps(
        self,
        zarr_path: Union[str, Path],
        map_type: str = "species",
        species: Optional[List[str]] = None,
        output_dir: Union[str, Path] = "maps",
        format: str = "png",
        dpi: int = 300,
        cmap: Optional[str] = None,
        show_all: bool = False,
        state: Optional[str] = None,
        basemap: Optional[str] = None
    ) -> List[Path]:
        """
        Create maps from Zarr data.
        
        Parameters
        ----------
        zarr_path : str or Path
            Path to Zarr store.
        map_type : str, default="species"
            Type of map: "species", "diversity", "richness", "comparison".
        species : List[str], optional
            Species codes for species/comparison maps.
        output_dir : str or Path, default="maps"
            Output directory for maps.
        format : str, default="png"
            Output format.
        dpi : int, default=300
            Output resolution.
        cmap : str, optional
            Colormap name.
        show_all : bool, default=False
            Create maps for all species.
        state : str, optional
            State boundary to overlay.
        basemap : str, optional
            Basemap provider.
            
        Returns
        -------
        List[Path]
            Paths to created map files.
            
        Examples
        --------
        >>> api = GridFIA()
        >>> # Create species map
        >>> maps = api.create_maps(
        ...     "data/forest.zarr",
        ...     map_type="species",
        ...     species=["0202"],
        ...     state="MT"
        ... )
        >>> 
        >>> # Create diversity maps
        >>> maps = api.create_maps(
        ...     "data/forest.zarr",
        ...     map_type="diversity"
        ... )
        >>> 
        >>> # Create comparison map
        >>> maps = api.create_maps(
        ...     "data/forest.zarr",
        ...     map_type="comparison",
        ...     species=["0202", "0122", "0116"]
        ... )
        """
        zarr_path = Path(zarr_path)
        output_dir = Path(output_dir)
        
        if not zarr_path.exists():
            raise InvalidZarrStructure(
                f"Zarr store not found: {zarr_path}",
                zarr_path=str(zarr_path)
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize mapper
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
        
        created_maps = []
        
        if map_type == "species":
            if show_all:
                # Create maps for all species
                species_info = mapper.get_species_info()
                for sp in species_info:
                    if sp['code'] != '0000':  # Skip total biomass
                        fig, ax = mapper.create_species_map(
                            species=sp['index'],
                            cmap=cmap,
                            state_boundary=state,
                            basemap=basemap
                        )
                        
                        output_path = output_dir / f"species_{sp['code']}_{sp['name'].replace(' ', '_')}.{format}"
                        from .visualization.plots import save_figure
                        save_figure(fig, str(output_path), dpi=dpi)
                        created_maps.append(output_path)
                        
                        import matplotlib.pyplot as plt
                        plt.close(fig)
            
            elif species:
                # Create maps for specified species
                for sp_code in species:
                    fig, ax = mapper.create_species_map(
                        species=sp_code,
                        cmap=cmap,
                        state_boundary=state,
                        basemap=basemap
                    )
                    
                    output_path = output_dir / f"species_{sp_code}.{format}"
                    from .visualization.plots import save_figure
                    save_figure(fig, str(output_path), dpi=dpi)
                    created_maps.append(output_path)
                    
                    import matplotlib.pyplot as plt
                    plt.close(fig)
            else:
                raise SpeciesNotFound(
                    "Please specify species codes or use show_all=True"
                )
        
        elif map_type == "diversity":
            # Create diversity maps
            for div_type in ['shannon', 'simpson']:
                fig, ax = mapper.create_diversity_map(
                    diversity_type=div_type,
                    cmap=cmap,
                    state_boundary=state,
                    basemap=basemap
                )
                
                output_path = output_dir / f"{div_type}_diversity.{format}"
                from .visualization.plots import save_figure
                save_figure(fig, str(output_path), dpi=dpi)
                created_maps.append(output_path)
                
                import matplotlib.pyplot as plt
                plt.close(fig)
        
        elif map_type == "richness":
            # Create richness map
            fig, ax = mapper.create_richness_map(
                cmap=cmap,
                state_boundary=state,
                basemap=basemap
            )
            
            output_path = output_dir / f"species_richness.{format}"
            from .visualization.plots import save_figure
            save_figure(fig, str(output_path), dpi=dpi)
            created_maps.append(output_path)
            
            import matplotlib.pyplot as plt
            plt.close(fig)
        
        elif map_type == "comparison":
            # Create comparison map
            if not species or len(species) < 2:
                raise SpeciesNotFound(
                    "Comparison maps require at least 2 species",
                    available_species=species
                )
            
            fig = mapper.create_comparison_map(
                species_list=species,
                cmap=cmap
            )
            
            output_path = output_dir / f"species_comparison.{format}"
            from .visualization.plots import save_figure
            save_figure(fig, str(output_path), dpi=dpi)
            created_maps.append(output_path)
            
            import matplotlib.pyplot as plt
            plt.close(fig)
        
        else:
            raise CalculationFailed(
                f"Unknown map type: {map_type}. Valid types: species, diversity, richness, comparison",
                calculation_name=map_type,
                available_calculations=["species", "diversity", "richness", "comparison"]
            )
        
        logger.info(f"Created {len(created_maps)} maps in {output_dir}")
        return created_maps
    
    def get_location_config(
        self,
        state: Optional[str] = None,
        county: Optional[str] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        crs: str = "EPSG:4326",
        output_path: Optional[Union[str, Path]] = None
    ) -> LocationConfig:
        """
        Create or retrieve location configuration.
        
        Parameters
        ----------
        state : str, optional
            State name or abbreviation.
        county : str, optional
            County name (requires state).
        bbox : Tuple[float, float, float, float], optional
            Custom bounding box.
        crs : str, default="EPSG:4326"
            CRS for custom bbox.
        output_path : str or Path, optional
            Path to save configuration.
            
        Returns
        -------
        LocationConfig
            Location configuration object.
            
        Examples
        --------
        >>> api = GridFIA()
        >>> # Get state configuration
        >>> config = api.get_location_config(state="Montana")
        >>> print(f"Bbox: {config.web_mercator_bbox}")
        >>> 
        >>> # Get county configuration
        >>> config = api.get_location_config(state="Texas", county="Harris")
        >>> 
        >>> # Custom bbox configuration
        >>> config = api.get_location_config(
        ...     bbox=(-104, 44, -104.5, 44.5),
        ...     crs="EPSG:4326"
        ... )
        """
        if county and not state:
            raise InvalidLocationConfig(
                "County requires state to be specified",
                location_type="county",
                county=county
            )
        
        if bbox:
            config = LocationConfig.from_bbox(
                bbox,
                name="Custom Region",
                crs=crs,
                output_path=output_path
            )
        elif county:
            config = LocationConfig.from_county(
                county, state, output_path=output_path
            )
        elif state:
            config = LocationConfig.from_state(
                state, output_path=output_path
            )
        else:
            raise InvalidLocationConfig(
                "Must specify state, county, or bbox",
                location_type="unknown"
            )

        return config
    
    def list_calculations(self) -> List[str]:
        """
        List all available calculations.
        
        Returns
        -------
        List[str]
            Names of available calculations.
            
        Examples
        --------
        >>> api = GridFIA()
        >>> calcs = api.list_calculations()
        >>> print(f"Available calculations: {calcs}")
        """
        return registry.list_calculations()
    
    def validate_zarr(self, zarr_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate a Zarr store and return metadata.

        Parameters
        ----------
        zarr_path : str or Path
            Path to Zarr store.

        Returns
        -------
        Dict[str, Any]
            Zarr store metadata including shape, species, chunks, etc.

        Examples
        --------
        >>> api = GridFIA()
        >>> info = api.validate_zarr("data/forest.zarr")
        >>> print(f"Shape: {info['shape']}")
        >>> print(f"Species: {info['num_species']}")
        """
        return validate_zarr_store(Path(zarr_path))

    # Static metadata for sample datasets (URLs generated from config)
    _SAMPLE_METADATA = {
        "durham_nc": {
            "name": "Durham County, North Carolina",
            "description": "Full Durham County with all species (~263 MB)",
            "num_species": 326,
            "approximate_size_mb": 263
        },
    }

    # Static metadata for state datasets (URLs generated from config)
    _STATE_METADATA = {
        "RI": {
            "name": "Rhode Island",
            "description": "Full Rhode Island state coverage",
            "num_species": 326,
            "shape": [326, 3407, 2264],
            "approximate_size_mb": 646
        },
        "CT": {
            "name": "Connecticut",
            "description": "Full Connecticut state coverage",
            "num_species": 326,
            "shape": [326, 4100, 7151],
            "approximate_size_mb": 2807
        },
    }

    @property
    def SAMPLE_DATASETS(self) -> Dict[str, Dict[str, Any]]:
        """Get sample datasets with URLs from cloud config."""
        return {
            key: {
                **meta,
                "url": self.settings.cloud.get_sample_url(key)
            }
            for key, meta in self._SAMPLE_METADATA.items()
        }

    @property
    def STATE_DATASETS(self) -> Dict[str, Dict[str, Any]]:
        """Get state datasets with URLs from cloud config."""
        return {
            key: {
                **meta,
                "url": self.settings.cloud.get_state_url(key)
            }
            for key, meta in self._STATE_METADATA.items()
        }

    def list_sample_datasets(self) -> List[Dict[str, Any]]:
        """
        List available pre-hosted sample datasets.

        These datasets are hosted on cloud storage and can be loaded
        instantly without downloading from the FIA API.

        Returns
        -------
        List[Dict[str, Any]]
            List of available sample datasets with metadata.

        Examples
        --------
        >>> api = GridFIA()
        >>> samples = api.list_sample_datasets()
        >>> for s in samples:
        ...     print(f"{s['key']}: {s['name']} ({s['approximate_size_mb']} MB)")
        """
        return [
            {"key": key, **info}
            for key, info in self.SAMPLE_DATASETS.items()
        ]

    def list_state_datasets(self) -> List[Dict[str, Any]]:
        """
        List available pre-hosted state datasets.

        These are full-state forest data hosted on cloud storage, enabling
        streaming access to any US state's forest data without local download.

        Returns
        -------
        List[Dict[str, Any]]
            List of available state datasets with metadata.

        Examples
        --------
        >>> api = GridFIA()
        >>> states = api.list_state_datasets()
        >>> for s in states:
        ...     print(f"{s['state']}: {s['name']} ({s['approximate_size_mb']} MB)")
        """
        return [
            {"state": key, **info}
            for key, info in self.STATE_DATASETS.items()
        ]

    def load_state(
        self,
        state: str,
        storage_options: Optional[Dict[str, Any]] = None
    ) -> ZarrStore:
        """
        Load a state's forest data from cloud storage.

        This enables streaming access to full-state forest data. Only the chunks
        you access are downloaded, making it efficient to analyze specific regions
        within a state.

        The cloud storage backend is configured via settings.cloud or environment
        variables (GRIDFIA_CLOUD_PUBLIC_URL, etc.). See CloudStorageConfig for details.

        Parameters
        ----------
        state : str
            State abbreviation (e.g., "NC", "CA", "RI").
        storage_options : Dict[str, Any], optional
            Options passed to the filesystem backend. If None, uses settings
            from cloud configuration.

        Returns
        -------
        ZarrStore
            A ZarrStore instance for streaming access to the state data.

        Raises
        ------
        ValueError
            If the state is not available in cloud storage.

        Examples
        --------
        >>> api = GridFIA()
        >>> store = api.load_state("RI")
        >>> print(f"Shape: {store.shape}")
        >>> print(f"Species: {store.num_species}")

        >>> # With custom cloud config
        >>> from gridfia.config import CloudStorageConfig
        >>> cloud = CloudStorageConfig(
        ...     public_url="https://f004.backblazeb2.com/file/my-bucket"
        ... )
        >>> api = GridFIA()
        >>> api.settings.cloud = cloud
        >>> store = api.load_state("RI")
        """
        state_upper = state.upper()
        if state_upper not in self._STATE_METADATA:
            available = list(self._STATE_METADATA.keys())
            raise ValueError(
                f"State '{state}' not available. Available states: {available}"
            )

        url = self.settings.cloud.get_state_url(state_upper)

        # Merge config storage options with user-provided options
        # Pass URL to get correct options (HTTP vs S3)
        config_options = self.settings.cloud.get_storage_options(url=url)
        if storage_options:
            config_options.update(storage_options)

        return self.load_from_cloud(url=url, storage_options=config_options)

    def load_from_cloud(
        self,
        url: Optional[str] = None,
        sample: Optional[str] = None,
        storage_options: Optional[Dict[str, Any]] = None
    ) -> ZarrStore:
        """
        Load a Zarr store from cloud storage for streaming access.

        This enables efficient access to cloud-hosted forest data. Only the
        chunks you access are downloaded, making it efficient to work with
        subsets of large datasets (e.g., one county from a US-wide store).

        Parameters
        ----------
        url : str, optional
            Direct URL to a Zarr store. Supports HTTP, S3, R2, GCS.
        sample : str, optional
            Name of a pre-hosted sample dataset (e.g., "durham_nc").
            See list_sample_datasets() for available options.
        storage_options : Dict[str, Any], optional
            Options passed to the filesystem backend.

        Returns
        -------
        ZarrStore
            A ZarrStore instance for streaming access to the data.

        Raises
        ------
        ValueError
            If neither url nor sample is provided, or if sample is unknown.
        ImportError
            If fsspec is not installed.

        Examples
        --------
        >>> api = GridFIA()
        >>>
        >>> # Load a pre-hosted sample dataset
        >>> store = api.load_from_cloud(sample="durham_nc")
        >>> print(f"Species: {store.num_species}")
        >>> print(f"Shape: {store.shape}")
        >>>
        >>> # Load from custom URL
        >>> store = api.load_from_cloud(
        ...     url="https://your-bucket.r2.dev/forest_data.zarr"
        ... )
        >>>
        >>> # Access data - only downloads needed chunks
        >>> biomass = store.biomass[:, 100:200, 100:200]  # Small subset
        >>>
        >>> # Use with calculations (works like local Zarr)
        >>> # Note: For calculations, you may want to download to local first
        >>> # for better performance with large operations

        Notes
        -----
        For best performance:
        - Use pre-hosted samples for demos and tutorials
        - For production analysis, consider downloading to local Zarr first
        - Remote stores work best for exploratory analysis of subsets
        """
        if sample:
            if sample not in self._SAMPLE_METADATA:
                available = list(self._SAMPLE_METADATA.keys())
                raise ValueError(
                    f"Unknown sample dataset: '{sample}'. "
                    f"Available: {available}"
                )
            url = self.settings.cloud.get_sample_url(sample)
            logger.info(f"Loading sample dataset: {self._SAMPLE_METADATA[sample]['name']}")

        if not url:
            raise ValueError(
                "Must provide either 'url' or 'sample' parameter. "
                "Use list_sample_datasets() to see available samples."
            )

        # Merge config storage options with user-provided options
        # Pass URL to get correct options (HTTP vs S3)
        config_options = self.settings.cloud.get_storage_options(url=url)
        if storage_options:
            config_options.update(storage_options)

        logger.info(f"Opening cloud Zarr store: {url}")
        return ZarrStore.from_url(url, storage_options=config_options)

    def download_sample(
        self,
        sample: str,
        output_path: Union[str, Path],
        show_progress: bool = True
    ) -> Path:
        """
        Download a sample dataset to local storage.

        Use this when you need faster repeated access or want to run
        calculations on the full dataset.

        Parameters
        ----------
        sample : str
            Name of the sample dataset (e.g., "durham_nc").
        output_path : str or Path
            Local path to save the Zarr store.
        show_progress : bool, default=True
            Whether to show download progress.

        Returns
        -------
        Path
            Path to the downloaded Zarr store.

        Examples
        --------
        >>> api = GridFIA()
        >>> local_path = api.download_sample("durham_nc", "data/durham.zarr")
        >>> results = api.calculate_metrics(local_path)
        """
        import shutil
        import urllib.request

        if sample not in self._SAMPLE_METADATA:
            available = list(self._SAMPLE_METADATA.keys())
            raise ValueError(
                f"Unknown sample dataset: '{sample}'. Available: {available}"
            )

        output_path = Path(output_path)
        dataset_info = self._SAMPLE_METADATA[sample]

        logger.info(f"Downloading {dataset_info['name']} to {output_path}")

        # For now, use the cloud loading and copy approach
        # A more efficient implementation would use streaming download
        store = self.load_from_cloud(sample=sample)

        # Copy to local zarr
        output_path.mkdir(parents=True, exist_ok=True)

        import zarr
        local_store = zarr.storage.LocalStore(output_path)
        local_root = zarr.open_group(store=local_store, mode='w')

        # Copy arrays and attributes
        local_root.attrs.update(store.attrs)

        # Copy biomass array - Zarr v3 doesn't allow both data and dtype
        import numpy as np
        biomass_data = np.array(store.biomass[:], dtype=store.dtype)
        local_root.create_array('biomass', data=biomass_data, chunks=store.chunks)

        # Copy species metadata - convert to numpy arrays for Zarr v3 compatibility
        codes_array = np.array(store.species_codes, dtype='U10')
        names_array = np.array(store.species_names, dtype='U100')
        local_root.create_array('species_codes', data=codes_array)
        local_root.create_array('species_names', data=names_array)

        logger.info(f"Downloaded to {output_path}")
        return output_path