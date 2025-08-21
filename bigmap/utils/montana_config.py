"""
Configuration loader for Montana Forest Analysis project.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from rich.console import Console

console = Console()


class MontanaConfig:
    """Configuration manager for Montana Forest Analysis project."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration from YAML file."""
        if config_path is None:
            config_path = Path("config/montana_project.yml")
        
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        console.print(f"[green]Loaded Montana configuration from:[/green] {self.config_path}")
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration value by key."""
        return self._config[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return self._config.get(key, default)
    
    @property
    def project_name(self) -> str:
        """Get project name."""
        return self._config['project']['name']
    
    @property
    def target_crs(self) -> str:
        """Get target CRS (EPSG:2256)."""
        return self._config['crs']['target']
    
    @property
    def state_plane_bbox(self) -> Tuple[float, float, float, float]:
        """Get Montana bounding box in State Plane coordinates as tuple."""
        bbox = self._config['bounding_boxes']['state_plane']
        return (bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax'])
    
    @property
    def state_plane_bbox_dict(self) -> Dict[str, float]:
        """Get Montana bounding box in State Plane coordinates as dict."""
        return self._config['bounding_boxes']['state_plane']
    
    @property
    def wgs84_bbox(self) -> Tuple[float, float, float, float]:
        """Get Montana bounding box in WGS84 coordinates as tuple."""
        bbox = self._config['bounding_boxes']['wgs84']
        return (bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax'])
    
    @property
    def web_mercator_bbox(self) -> Tuple[float, float, float, float]:
        """Get Montana bounding box in Web Mercator coordinates as tuple."""
        bbox = self._config['bounding_boxes']['web_mercator']
        return (bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax'])
    
    @property
    def species_list(self) -> list:
        """Get list of species."""
        return self._config['species']
    
    @property
    def species_codes(self) -> list:
        """Get list of species codes."""
        return [s['code'] for s in self.species_list]
    
    @property
    def species_names(self) -> list:
        """Get list of species names."""
        return [s['name'] for s in self.species_list]
    
    @property
    def zarr_output_path(self) -> Path:
        """Get zarr output path."""
        return Path(self._config['zarr']['output_path'])
    
    @property
    def zarr_layers(self) -> list:
        """Get zarr layer configuration."""
        return self._config['zarr']['layers']
    
    @property
    def chunk_size(self) -> Tuple[int, int, int]:
        """Get zarr chunk size."""
        return tuple(self._config['zarr']['chunk_size'])
    
    @property
    def compression(self) -> str:
        """Get compression algorithm."""
        return self._config['zarr']['compression']
    
    @property
    def compression_level(self) -> int:
        """Get compression level."""
        return self._config['zarr']['compression_level']
    
    @property
    def download_resolution_ft(self) -> float:
        """Get download resolution in feet."""
        return self._config['download']['resolution_ft']
    
    @property
    def download_output_dir(self) -> Path:
        """Get download output directory."""
        return Path(self._config['download']['output_dir'])
    
    @property
    def county_shapefile(self) -> Path:
        """Get county shapefile path."""
        return Path(self._config['counties']['shapefile'])
    
    @property
    def county_output_raster(self) -> Path:
        """Get county output raster path."""
        return Path(self._config['counties']['output_raster'])
    
    @property
    def state_fips(self) -> str:
        """Get Montana state FIPS code."""
        return self._config['counties']['state_fips']
    
    @property
    def layer_indices(self) -> Dict[str, int]:
        """Get layer indices mapping."""
        return self._config['zarr']['layer_indices']
    
    @property
    def species_start_idx(self) -> int:
        """Get starting index for species layers."""
        return self.layer_indices['species_start']
    
    @property
    def species_end_idx(self) -> int:
        """Get ending index for species layers (inclusive)."""
        return self.layer_indices['species_end']
    
    @property
    def timber_idx(self) -> int:
        """Get timber layer index."""
        return self.layer_indices['timber']
    
    @property
    def dominant_species_idx(self) -> int:
        """Get dominant species layer index."""
        return self.layer_indices['dominant_species']
    
    @property
    def maps_dir(self) -> Path:
        """Get maps output directory."""
        return Path(self._config['paths']['maps_dir'])
    
    @property
    def data_dir(self) -> Path:
        """Get data directory."""
        return Path(self._config['paths']['data_dir'])
    
    def get_species_by_code(self, code: str) -> Optional[Dict[str, str]]:
        """Get species information by code."""
        for species in self.species_list:
            if species['code'] == code:
                return species
        return None
    
    def get_layer_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get layer information by name."""
        for layer in self.zarr_layers:
            if layer['name'].lower() == name.lower():
                return layer
        return None
    
    def get_layer_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """Get layer information by index."""
        for layer in self.zarr_layers:
            if layer['index'] == index:
                return layer
        return None
    
    def print_summary(self):
        """Print configuration summary."""
        console.print("\n[bold cyan]Montana Project Configuration:[/bold cyan]")
        console.print(f"Project: {self.project_name}")
        console.print(f"Target CRS: {self.target_crs} (NAD83 / Montana State Plane)")
        
        console.print(f"\n[cyan]State Plane Bounding Box:[/cyan]")
        bbox = self.state_plane_bbox_dict
        console.print(f"  X: {bbox['xmin']:,.2f} to {bbox['xmax']:,.2f} ft")
        console.print(f"  Y: {bbox['ymin']:,.2f} to {bbox['ymax']:,.2f} ft")
        width_ft = bbox['xmax'] - bbox['xmin']
        height_ft = bbox['ymax'] - bbox['ymin']
        console.print(f"  Width: {width_ft:,.2f} ft ({width_ft * 0.3048 / 1000:,.2f} km)")
        console.print(f"  Height: {height_ft:,.2f} ft ({height_ft * 0.3048 / 1000:,.2f} km)")
        
        console.print(f"\n[cyan]Species ({len(self.species_list)}):[/cyan]")
        for species in self.species_list:
            console.print(f"  - [{species['code']}] {species['name']} ({species['scientific_name']})")
        
        console.print(f"\n[cyan]Zarr Layers ({len(self.zarr_layers)}):[/cyan]")
        for layer in self.zarr_layers:
            console.print(f"  {layer['index']}: {layer['name']} [{layer['code']}]")
        
        console.print(f"\n[cyan]Output Paths:[/cyan]")
        console.print(f"  Zarr store: {self.zarr_output_path}")
        console.print(f"  Downloads: {self.download_output_dir}")
        console.print(f"  County raster: {self.county_output_raster}")


# Global instance
_montana_config: Optional[MontanaConfig] = None


def load_montana_config(config_path: Optional[Path] = None) -> MontanaConfig:
    """Load Montana configuration."""
    global _montana_config
    _montana_config = MontanaConfig(config_path)
    return _montana_config


def get_montana_config() -> MontanaConfig:
    """Get current Montana configuration."""
    global _montana_config
    if _montana_config is None:
        _montana_config = load_montana_config()
    return _montana_config