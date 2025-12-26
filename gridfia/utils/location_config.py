"""
Generic location configuration for any US state or county.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
from rich.console import Console
import geopandas as gpd
from shapely.geometry import box
from rasterio.crs import CRS
from rasterio.warp import transform_bounds

from ..exceptions import InvalidLocationConfig

console = Console()


class LocationConfig:
    """Configuration manager for any geographic location (state, county, custom region)."""
    
    def __init__(self, config_path: Optional[Path] = None, location_type: str = "state"):
        """
        Initialize configuration from YAML file or create from location.
        
        Args:
            config_path: Path to configuration YAML file
            location_type: Type of location ("state", "county", "custom")
        """
        self._location_type = location_type
        
        if config_path:
            self.config_path = Path(config_path)
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
            
            console.print(f"[green]Loaded {location_type} configuration from:[/green] {self.config_path}")
        else:
            self._config = self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create a default configuration template."""
        return {
            'project': {
                'name': "Forest Biomass Analysis",
                'description': "Forest biomass and diversity analysis",
                'version': "1.0.0"
            },
            'location': {
                'type': self._location_type,
                'name': None,
                'abbreviation': None,
                'fips_code': None
            },
            'crs': {
                'source': "EPSG:4326",
                'target': None,
                'web_mercator': "EPSG:3857"
            },
            'bounding_boxes': {
                'wgs84': None,
                'state_plane': None,
                'web_mercator': None
            },
            'species': [],
            'zarr': {
                'output_path': "output/data/forest_biomass.zarr",
                'chunk_size': [1, 1000, 1000],
                'compression': 'lz4',
                'compression_level': 5
            },
            'download': {
                'resolution_ft': 98.425197,
                'output_dir': "output/data/species",
                'max_retries': 3,
                'timeout': 60,
                'rate_limit_delay': 0.5
            },
            'visualization': {
                'biomass_cmap': 'YlGn',
                'diversity_cmap': 'plasma',
                'richness_cmap': 'Spectral_r',
                'boundary_color': 'black',
                'boundary_linewidth': 0.5,
                'figure_size': [12, 10],
                'dpi': 150
            },
            'analysis': {
                'presence_threshold': 1.0,
                'normalization_percentiles': [2, 98],
                'nodata_value': -9999
            },
            'paths': {
                'data_dir': "output/data",
                'maps_dir': "output/maps",
                'scripts_dir': "scripts",
                'examples_dir': "examples"
            }
        }
    
    @classmethod
    def from_state(cls, state: str, output_path: Optional[Path] = None) -> 'LocationConfig':
        """
        Create configuration for a specific US state.
        
        Args:
            state: State name or abbreviation
            output_path: Path to save configuration file
            
        Returns:
            LocationConfig instance for the state
        """
        config = cls(location_type="state")
        config._setup_state_config(state)
        
        if output_path:
            config.save(output_path)
        
        return config
    
    @classmethod
    def from_county(cls, county: str, state: str, output_path: Optional[Path] = None) -> 'LocationConfig':
        """
        Create configuration for a specific county.
        
        Args:
            county: County name
            state: State name or abbreviation
            output_path: Path to save configuration file
            
        Returns:
            LocationConfig instance for the county
        """
        config = cls(location_type="county")
        config._setup_county_config(county, state)
        
        if output_path:
            config.save(output_path)
        
        return config
    
    @classmethod
    def from_bbox(cls, bbox: Tuple[float, float, float, float], 
                  name: str = "Custom Region",
                  crs: str = "EPSG:4326",
                  output_path: Optional[Path] = None) -> 'LocationConfig':
        """
        Create configuration for a custom bounding box.
        
        Args:
            bbox: Bounding box (xmin, ymin, xmax, ymax)
            name: Name for the region
            crs: CRS of the bounding box
            output_path: Path to save configuration file
            
        Returns:
            LocationConfig instance for the custom region
        """
        config = cls(location_type="custom")
        config._setup_custom_config(bbox, name, crs)
        
        if output_path:
            config.save(output_path)
        
        return config
    
    def _setup_state_config(self, state: str):
        """Setup configuration for a US state."""
        from gridfia.visualization.boundaries import load_state_boundary, STATE_ABBR
        
        state_lower = state.lower()
        if state_lower in STATE_ABBR:
            state_abbr = STATE_ABBR[state_lower]
            state_name = state_lower.title()
        else:
            state_abbr = state.upper()
            state_name = None
            for name, abbr in STATE_ABBR.items():
                if abbr == state_abbr:
                    state_name = name.title()
                    break
        
        if not state_name:
            raise InvalidLocationConfig(
                f"Unknown state: {state}",
                location_type="state",
                state=state
            )

        self._config['location']['name'] = state_name
        self._config['location']['abbreviation'] = state_abbr

        try:
            gdf = load_state_boundary(state)
            self._setup_bounding_boxes(gdf)
            self._detect_state_plane_crs(state_abbr)
        except InvalidLocationConfig:
            raise
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load boundaries for {state_name}: {e}[/yellow]")

    def _setup_county_config(self, county: str, state: str):
        """Setup configuration for a county."""
        from gridfia.visualization.boundaries import load_counties_for_state, STATE_ABBR
        
        state_lower = state.lower()
        if state_lower in STATE_ABBR:
            state_name = state_lower.title()
        else:
            state_abbr = state.upper()
            state_name = None
            for name, abbr in STATE_ABBR.items():
                if abbr == state_abbr:
                    state_name = name.title()
                    break
        
        if not state_name:
            raise InvalidLocationConfig(
                f"Unknown state: {state}",
                location_type="county",
                state=state,
                county=county
            )

        self._config['location']['name'] = f"{county} County, {state_name}"
        self._config['location']['state'] = state_name
        self._config['location']['county'] = county

        try:
            counties_gdf = load_counties_for_state(state)
            county_gdf = counties_gdf[counties_gdf['NAME'].str.lower() == county.lower()]

            if county_gdf.empty:
                raise InvalidLocationConfig(
                    f"County {county} not found in {state_name}",
                    location_type="county",
                    state=state_name,
                    county=county
                )

            self._setup_bounding_boxes(county_gdf)
            self._detect_state_plane_crs(state.upper() if len(state) == 2 else STATE_ABBR.get(state.lower()))
        except InvalidLocationConfig:
            raise
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load boundaries for {county}, {state_name}: {e}[/yellow]")
    
    def _setup_custom_config(self, bbox: Tuple[float, float, float, float], name: str, crs: str):
        """Setup configuration for a custom bounding box."""
        self._config['location']['name'] = name
        
        if crs == "EPSG:4326":
            self._config['bounding_boxes']['wgs84'] = {
                'xmin': bbox[0], 'ymin': bbox[1],
                'xmax': bbox[2], 'ymax': bbox[3]
            }
        elif crs == "EPSG:3857":
            self._config['bounding_boxes']['web_mercator'] = {
                'xmin': bbox[0], 'ymin': bbox[1],
                'xmax': bbox[2], 'ymax': bbox[3]
            }
        else:
            self._config['bounding_boxes']['state_plane'] = {
                'xmin': bbox[0], 'ymin': bbox[1],
                'xmax': bbox[2], 'ymax': bbox[3]
            }
            self._config['crs']['target'] = crs
        
        self._convert_bounding_boxes()
    
    def _setup_bounding_boxes(self, gdf: gpd.GeoDataFrame):
        """Setup bounding boxes from a GeoDataFrame."""
        bounds = gdf.total_bounds
        
        original_crs = gdf.crs
        
        gdf_wgs84 = gdf.to_crs("EPSG:4326")
        bounds_wgs84 = gdf_wgs84.total_bounds
        self._config['bounding_boxes']['wgs84'] = {
            'xmin': float(bounds_wgs84[0]), 'ymin': float(bounds_wgs84[1]),
            'xmax': float(bounds_wgs84[2]), 'ymax': float(bounds_wgs84[3])
        }
        
        gdf_mercator = gdf.to_crs("EPSG:3857")
        bounds_mercator = gdf_mercator.total_bounds
        self._config['bounding_boxes']['web_mercator'] = {
            'xmin': float(bounds_mercator[0]), 'ymin': float(bounds_mercator[1]),
            'xmax': float(bounds_mercator[2]), 'ymax': float(bounds_mercator[3])
        }
        
        if original_crs and str(original_crs) not in ["EPSG:4326", "EPSG:3857"]:
            self._config['bounding_boxes']['state_plane'] = {
                'xmin': float(bounds[0]), 'ymin': float(bounds[1]),
                'xmax': float(bounds[2]), 'ymax': float(bounds[3])
            }
            self._config['crs']['target'] = str(original_crs)
    
    def _convert_bounding_boxes(self):
        """Convert between different CRS for bounding boxes."""
        if self._config['bounding_boxes']['wgs84']:
            wgs_bbox = self._config['bounding_boxes']['wgs84']
            bounds = (wgs_bbox['xmin'], wgs_bbox['ymin'], wgs_bbox['xmax'], wgs_bbox['ymax'])
            
            mercator_bounds = transform_bounds("EPSG:4326", "EPSG:3857", *bounds)
            self._config['bounding_boxes']['web_mercator'] = {
                'xmin': mercator_bounds[0], 'ymin': mercator_bounds[1],
                'xmax': mercator_bounds[2], 'ymax': mercator_bounds[3]
            }
            
            if self._config['crs']['target']:
                sp_bounds = transform_bounds("EPSG:4326", self._config['crs']['target'], *bounds)
                self._config['bounding_boxes']['state_plane'] = {
                    'xmin': sp_bounds[0], 'ymin': sp_bounds[1],
                    'xmax': sp_bounds[2], 'ymax': sp_bounds[3]
                }
    
    def _detect_state_plane_crs(self, state_abbr: str):
        """Detect the appropriate State Plane CRS for a state."""
        STATE_PLANE_CRS = {
            'AL': 'EPSG:26929',  # Alabama East
            'AK': 'EPSG:26931',  # Alaska Zone 1
            'AZ': 'EPSG:26948',  # Arizona Central
            'AR': 'EPSG:26951',  # Arkansas North
            'CA': 'EPSG:26943',  # California Zone III
            'CO': 'EPSG:26953',  # Colorado Central
            'CT': 'EPSG:26956',  # Connecticut
            'DE': 'EPSG:26957',  # Delaware
            'FL': 'EPSG:26958',  # Florida East
            'GA': 'EPSG:26966',  # Georgia East
            'HI': 'EPSG:26961',  # Hawaii Zone 1
            'ID': 'EPSG:26968',  # Idaho Central
            'IL': 'EPSG:26971',  # Illinois East
            'IN': 'EPSG:26973',  # Indiana East
            'IA': 'EPSG:26975',  # Iowa North
            'KS': 'EPSG:26977',  # Kansas North
            'KY': 'EPSG:26979',  # Kentucky North
            'LA': 'EPSG:26981',  # Louisiana North
            'ME': 'EPSG:26983',  # Maine East
            'MD': 'EPSG:26985',  # Maryland
            'MA': 'EPSG:26986',  # Massachusetts Mainland
            'MI': 'EPSG:26988',  # Michigan Central
            'MN': 'EPSG:26991',  # Minnesota Central
            'MS': 'EPSG:26994',  # Mississippi East
            'MO': 'EPSG:26996',  # Missouri Central
            'MT': 'EPSG:2256',   # Montana State Plane
            'NE': 'EPSG:26992',  # Nebraska
            'NV': 'EPSG:26997',  # Nevada Central
            'NH': 'EPSG:26955',  # New Hampshire
            'NJ': 'EPSG:26954',  # New Jersey
            'NM': 'EPSG:26913',  # New Mexico Central
            'NY': 'EPSG:26918',  # New York Central
            'NC': 'EPSG:2264',   # North Carolina State Plane
            'ND': 'EPSG:2265',   # North Dakota North
            'OH': 'EPSG:26917',  # Ohio North
            'OK': 'EPSG:26914',  # Oklahoma North
            'OR': 'EPSG:26910',  # Oregon North
            'PA': 'EPSG:26918',  # Pennsylvania North
            'RI': 'EPSG:26919',  # Rhode Island
            'SC': 'EPSG:26919',  # South Carolina
            'SD': 'EPSG:26914',  # South Dakota North
            'TN': 'EPSG:26916',  # Tennessee
            'TX': 'EPSG:26914',  # Texas Central
            'UT': 'EPSG:26912',  # Utah Central
            'VT': 'EPSG:26919',  # Vermont
            'VA': 'EPSG:26918',  # Virginia North
            'WA': 'EPSG:26910',  # Washington North
            'WV': 'EPSG:26917',  # West Virginia North
            'WI': 'EPSG:26916',  # Wisconsin Central
            'WY': 'EPSG:26913'   # Wyoming East Central
        }
        
        if state_abbr in STATE_PLANE_CRS:
            self._config['crs']['target'] = STATE_PLANE_CRS[state_abbr]
        else:
            console.print(f"[yellow]Warning: No State Plane CRS found for {state_abbr}, using Web Mercator[/yellow]")
            self._config['crs']['target'] = "EPSG:3857"
    
    def save(self, output_path: Path):
        """Save configuration to YAML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
        
        console.print(f"[green]Saved configuration to:[/green] {output_path}")
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration value by key."""
        return self._config[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return self._config.get(key, default)
    
    @property
    def location_name(self) -> str:
        """Get location name."""
        return self._config['location']['name']
    
    @property
    def location_type(self) -> str:
        """Get location type."""
        return self._config.get('location', {}).get('type', 'state')
    
    @property
    def target_crs(self) -> str:
        """Get target CRS."""
        return self._config['crs']['target']
    
    @property
    def wgs84_bbox(self) -> Optional[Tuple[float, float, float, float]]:
        """Get WGS84 bounding box as tuple."""
        bbox = self._config['bounding_boxes'].get('wgs84')
        if bbox:
            return (bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax'])
        return None
    
    @property
    def web_mercator_bbox(self) -> Optional[Tuple[float, float, float, float]]:
        """Get Web Mercator bounding box as tuple."""
        bbox = self._config['bounding_boxes'].get('web_mercator')
        if bbox:
            return (bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax'])
        return None
    
    @property
    def state_plane_bbox(self) -> Optional[Tuple[float, float, float, float]]:
        """Get State Plane bounding box as tuple."""
        bbox = self._config['bounding_boxes'].get('state_plane')
        if bbox:
            return (bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax'])
        return None
    
    @property
    def species_list(self) -> List[Dict[str, str]]:
        """Get list of species."""
        return self._config.get('species', [])
    
    @property
    def zarr_output_path(self) -> Path:
        """Get zarr output path."""
        return Path(self._config['zarr']['output_path'])
    
    @property
    def chunk_size(self) -> Tuple[int, int, int]:
        """Get zarr chunk size."""
        return tuple(self._config['zarr']['chunk_size'])
    
    @property
    def compression(self) -> str:
        """Get compression algorithm."""
        return self._config['zarr']['compression']
    
    @property
    def download_output_dir(self) -> Path:
        """Get download output directory."""
        return Path(self._config['download']['output_dir'])
    
    def print_summary(self):
        """Print configuration summary."""
        console.print(f"\n[bold cyan]{self.location_name} Configuration:[/bold cyan]")
        console.print(f"Location Type: {self.location_type}")
        
        if self.target_crs:
            console.print(f"Target CRS: {self.target_crs}")
        
        if self.wgs84_bbox:
            bbox = self._config['bounding_boxes']['wgs84']
            console.print(f"\n[cyan]WGS84 Bounding Box:[/cyan]")
            console.print(f"  Longitude: {bbox['xmin']:.6f} to {bbox['xmax']:.6f}")
            console.print(f"  Latitude: {bbox['ymin']:.6f} to {bbox['ymax']:.6f}")
        
        if self.state_plane_bbox and self.target_crs:
            bbox = self._config['bounding_boxes']['state_plane']
            console.print(f"\n[cyan]State Plane Bounding Box:[/cyan]")
            console.print(f"  X: {bbox['xmin']:,.2f} to {bbox['xmax']:,.2f}")
            console.print(f"  Y: {bbox['ymin']:,.2f} to {bbox['ymax']:,.2f}")
        
        if self.species_list:
            console.print(f"\n[cyan]Species ({len(self.species_list)}):[/cyan]")
            for species in self.species_list:
                console.print(f"  - [{species['code']}] {species['name']}")
        
        console.print(f"\n[cyan]Output Paths:[/cyan]")
        console.print(f"  Zarr store: {self.zarr_output_path}")
        console.print(f"  Downloads: {self.download_output_dir}")


_location_config: Optional[LocationConfig] = None


def load_location_config(config_path: Optional[Path] = None) -> LocationConfig:
    """Load location configuration."""
    global _location_config
    _location_config = LocationConfig(config_path)
    return _location_config


def get_location_config() -> LocationConfig:
    """Get current location configuration."""
    global _location_config
    if _location_config is None:
        _location_config = LocationConfig()
    return _location_config