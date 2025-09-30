"""
BigMap API - Clean programmatic interface for forest biomass analysis.

This module provides the primary API for BigMap functionality, offering a clean,
well-documented interface for programmatic access to all features.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

import numpy as np
import xarray as xr
from pydantic import BaseModel, Field

from .config import BigMapSettings, CalculationConfig, load_settings
from .core.processors.forest_metrics import ForestMetricsProcessor
from .external.fia_client import BigMapRestClient
from .utils.location_config import LocationConfig
from .utils.zarr_utils import create_zarr_from_geotiffs, validate_zarr_store
from .visualization.mapper import ZarrMapper
from .core.calculations import registry

logger = logging.getLogger(__name__)


class CalculationResult(BaseModel):
    """Result from a calculation operation."""
    name: str
    output_path: Path
    statistics: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SpeciesInfo(BaseModel):
    """Information about a tree species."""
    species_code: str
    common_name: str
    scientific_name: str
    function_name: Optional[str] = None


class BigMapAPI:
    """
    Main API interface for BigMap forest analysis.
    
    This class provides a clean, programmatic interface to all BigMap functionality
    including data download, processing, analysis, and visualization.
    
    Examples
    --------
    >>> from bigmap import BigMapAPI
    >>> api = BigMapAPI()
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
    
    def __init__(self, config: Optional[Union[str, Path, BigMapSettings]] = None):
        """
        Initialize BigMap API.
        
        Parameters
        ----------
        config : str, Path, or BigMapSettings, optional
            Configuration file path or settings object.
            If None, uses default settings.
        """
        if config is None:
            self.settings = BigMapSettings()
        elif isinstance(config, (str, Path)):
            self.settings = load_settings(Path(config))
        else:
            self.settings = config
            
        self._rest_client = None
        self._processor = None
        
    @property
    def rest_client(self) -> BigMapRestClient:
        """Lazy-load REST client for FIA BIGMAP service."""
        if self._rest_client is None:
            self._rest_client = BigMapRestClient()
        return self._rest_client
    
    @property
    def processor(self) -> ForestMetricsProcessor:
        """Lazy-load forest metrics processor."""
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
        >>> api = BigMapAPI()
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
        polygon: Optional[Union[str, Path, "gpd.GeoDataFrame"]] = None,
        location_config: Optional[Union[str, Path]] = None,
        use_boundary_clip: bool = False,
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
        polygon : str, Path, or GeoDataFrame, optional
            Polygon boundary to use (GeoJSON, Shapefile, or GeoDataFrame).
            Data will be downloaded for the polygon's bbox and clipped to the polygon.
        location_config : str or Path, optional
            Path to location configuration file.
        use_boundary_clip : bool, default=False
            If True and using state/county, stores actual boundary for clipping.
            Only affects state/county downloads, ignored for bbox/polygon.
        crs : str, default="102100"
            Coordinate reference system for bbox.

        Returns
        -------
        List[Path]
            Paths to downloaded files.

        Examples
        --------
        >>> api = BigMapAPI()
        >>> # Download for entire state
        >>> files = api.download_species(state="Montana", species_codes=["0202"])
        >>>
        >>> # Download for specific county with boundary clipping
        >>> files = api.download_species(
        ...     state="Texas",
        ...     county="Harris",
        ...     species_codes=["0131", "0068"],
        ...     use_boundary_clip=True
        ... )
        >>>
        >>> # Download with custom bbox
        >>> files = api.download_species(
        ...     bbox=(-104, 44, -104.5, 44.5),
        ...     crs="4326"
        ... )
        >>>
        >>> # Download with custom polygon
        >>> files = api.download_species(
        ...     polygon="study_area.geojson",
        ...     species_codes=["0202"]
        ... )
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine location and bbox
        location_name = "location"
        location_bbox = None
        bbox_crs = crs
        location_config_obj = None

        if location_config:
            location_config_obj = LocationConfig(Path(location_config))
            location_name = location_config_obj.location_name.lower().replace(' ', '_')
            location_bbox = location_config_obj.web_mercator_bbox
            logger.info(f"Using location config: {location_config_obj.location_name}")

        elif polygon:
            # Create config from polygon
            location_config_obj = LocationConfig.from_polygon(polygon)
            location_name = location_config_obj.location_name.lower().replace(' ', '_')
            location_bbox = location_config_obj.web_mercator_bbox
            logger.info(f"Using polygon boundary: {location_config_obj.location_name}")

        elif state:
            if county:
                location_config_obj = LocationConfig.from_county(
                    county, state, store_boundary=use_boundary_clip
                )
                location_name = f"{county}_{state}".lower().replace(' ', '_')
            else:
                location_config_obj = LocationConfig.from_state(
                    state, store_boundary=use_boundary_clip
                )
                location_name = state.lower().replace(' ', '_')

            location_bbox = location_config_obj.web_mercator_bbox
            logger.info(f"Using {location_config_obj.location_name} boundaries")

        elif bbox:
            location_bbox = bbox

        else:
            raise ValueError("Must specify state, county, bbox, polygon, or location_config")

        if not location_bbox:
            raise ValueError("Could not determine bounding box for location")

        # Download species data
        exported_files = self.rest_client.batch_export_location_species(
            bbox=location_bbox,
            output_dir=output_dir,
            species_codes=species_codes,
            location_name=location_name,
            bbox_srs=bbox_crs
        )

        logger.info(f"Downloaded {len(exported_files)} species rasters")

        # Store location config for potential clipping
        if location_config_obj and location_config_obj.has_polygon:
            # Save config alongside downloads for later clipping
            config_path = output_dir / f"{location_name}_config.yaml"
            location_config_obj.save(config_path)
            logger.info(f"Saved location config with polygon boundary to {config_path}")

        return exported_files
    
    def create_zarr(
        self,
        input_dir: Union[str, Path],
        output_path: Union[str, Path],
        species_codes: Optional[List[str]] = None,
        chunk_size: Tuple[int, int, int] = (1, 1000, 1000),
        compression: str = "lz4",
        compression_level: int = 5,
        include_total: bool = True,
        clip_to_polygon: Optional[Union[bool, str, Path, "gpd.GeoDataFrame"]] = None
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
        clip_to_polygon : bool, str, Path, or GeoDataFrame, optional
            Polygon to clip rasters to before creating Zarr.
            - If True: looks for *_config.yaml in input_dir and uses its polygon
            - If str/Path: path to polygon file or config file
            - If GeoDataFrame: uses the provided polygon
            - If None/False: no clipping

        Returns
        -------
        Path
            Path to created Zarr store.

        Examples
        --------
        >>> api = BigMapAPI()
        >>> zarr_path = api.create_zarr(
        ...     "downloads/montana_species/",
        ...     "data/montana.zarr",
        ...     chunk_size=(1, 2000, 2000)
        ... )
        >>> print(f"Created Zarr store at {zarr_path}")
        >>>
        >>> # With polygon clipping
        >>> zarr_path = api.create_zarr(
        ...     "downloads/county_species/",
        ...     "data/county.zarr",
        ...     clip_to_polygon=True  # Auto-detect from config
        ... )
        """
        input_dir = Path(input_dir)
        output_path = Path(output_path)

        if not input_dir.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")

        # Handle polygon clipping
        polygon_gdf = None
        clipped_dir = None

        if clip_to_polygon:
            import geopandas as gpd
            from .utils.polygon_utils import clip_geotiffs_batch

            # Determine polygon source
            if isinstance(clip_to_polygon, bool) and clip_to_polygon:
                # Auto-detect config file in input_dir
                config_files = list(input_dir.glob("*_config.yaml"))
                if config_files:
                    config = LocationConfig(config_files[0])
                    if config.has_polygon:
                        polygon_gdf = config.polygon_gdf
                        logger.info(f"Using polygon from {config_files[0]}")
                    else:
                        logger.warning(f"Config file found but has no polygon boundary")
                else:
                    logger.warning(f"No config file found in {input_dir} for auto-clipping")

            elif isinstance(clip_to_polygon, (str, Path)):
                clip_path = Path(clip_to_polygon)
                if clip_path.suffix in ['.yaml', '.yml']:
                    # It's a config file
                    config = LocationConfig(clip_path)
                    if config.has_polygon:
                        polygon_gdf = config.polygon_gdf
                    else:
                        raise ValueError(f"Config file has no polygon boundary: {clip_path}")
                else:
                    # It's a polygon file
                    from .utils.polygon_utils import load_polygon
                    polygon_gdf = load_polygon(clip_path)

            elif isinstance(clip_to_polygon, gpd.GeoDataFrame):
                polygon_gdf = clip_to_polygon

            # Perform clipping if polygon available
            if polygon_gdf is not None:
                clipped_dir = input_dir / "clipped"
                logger.info(f"Clipping GeoTIFFs to polygon boundary...")
                clip_geotiffs_batch(
                    input_dir=input_dir,
                    polygon=polygon_gdf,
                    output_dir=clipped_dir,
                    pattern="*.tif*"
                )
                # Use clipped files as input
                input_dir = clipped_dir

        # Find GeoTIFF files
        tiff_files = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))

        if not tiff_files:
            raise ValueError(f"No GeoTIFF files found in {input_dir}")
        
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
                raise ValueError(f"No files found for species codes: {species_codes}")
        
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
        config: Optional[Union[str, Path, BigMapSettings]] = None
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
        config : str, Path, or BigMapSettings, optional
            Configuration to use for calculations.
            
        Returns
        -------
        List[CalculationResult]
            Results from each calculation.
            
        Examples
        --------
        >>> api = BigMapAPI()
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
            raise ValueError(f"Zarr store not found: {zarr_path}")
        
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
                raise ValueError(f"Unknown calculations: {invalid_calcs}. Available: {all_registered}")
            
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
        >>> api = BigMapAPI()
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
            raise ValueError(f"Zarr store not found: {zarr_path}")
        
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
                raise ValueError("Please specify species codes or use show_all=True")
        
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
                raise ValueError("Comparison maps require at least 2 species")
            
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
            raise ValueError(f"Unknown map type: {map_type}. Valid types: species, diversity, richness, comparison")
        
        logger.info(f"Created {len(created_maps)} maps in {output_dir}")
        return created_maps
    
    def get_location_config(
        self,
        state: Optional[str] = None,
        county: Optional[str] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        polygon: Optional[Union[str, Path, "gpd.GeoDataFrame"]] = None,
        store_boundary: bool = False,
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
        polygon : str, Path, or GeoDataFrame, optional
            Polygon boundary (GeoJSON, Shapefile, or GeoDataFrame).
        store_boundary : bool, default=False
            If True, stores actual boundary polygon for state/county.
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
        >>> api = BigMapAPI()
        >>> # Get state configuration
        >>> config = api.get_location_config(state="Montana")
        >>> print(f"Bbox: {config.web_mercator_bbox}")
        >>>
        >>> # Get county configuration with boundary
        >>> config = api.get_location_config(
        ...     state="Texas",
        ...     county="Harris",
        ...     store_boundary=True
        ... )
        >>>
        >>> # Custom bbox configuration
        >>> config = api.get_location_config(
        ...     bbox=(-104, 44, -104.5, 44.5),
        ...     crs="EPSG:4326"
        ... )
        >>>
        >>> # Polygon configuration
        >>> config = api.get_location_config(polygon="study_area.geojson")
        """
        if county and not state:
            raise ValueError("County requires state to be specified")

        if polygon:
            config = LocationConfig.from_polygon(
                polygon, output_path=output_path
            )
        elif bbox:
            config = LocationConfig.from_bbox(
                bbox,
                name="Custom Region",
                crs=crs,
                output_path=output_path
            )
        elif county:
            config = LocationConfig.from_county(
                county, state,
                store_boundary=store_boundary,
                output_path=output_path
            )
        elif state:
            config = LocationConfig.from_state(
                state,
                store_boundary=store_boundary,
                output_path=output_path
            )
        else:
            raise ValueError("Must specify state, county, bbox, or polygon")

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
        >>> api = BigMapAPI()
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
        >>> api = BigMapAPI()
        >>> info = api.validate_zarr("data/forest.zarr")
        >>> print(f"Shape: {info['shape']}")
        >>> print(f"Species: {info['num_species']}")
        """
        return validate_zarr_store(Path(zarr_path))