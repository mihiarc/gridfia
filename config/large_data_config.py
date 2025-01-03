"""Configuration settings for large data processing."""
from dataclasses import dataclass
from typing import Tuple, Dict
from pathlib import Path

@dataclass
class RasterConfig:
    """Configuration for raster processing."""
    chunk_size: Tuple[int, int] = (1024, 1024)  # Size of processing chunks
    compression: str = 'LZW'  # Compression algorithm for COGs
    block_size: Tuple[int, int] = (256, 256)  # Internal tiling size
    overview_levels: Tuple[int, ...] = (2, 4, 8, 16)  # Overview pyramid levels
    resampling: str = 'average'  # Resampling method for overviews

@dataclass
class VectorConfig:
    """Configuration for vector processing."""
    partition_size: int = 100000  # Number of rows per partition
    spatial_index: str = 'rtree'  # Spatial indexing method
    buffer_size: float = 1609.34  # Buffer size in meters (1 mile)
    chunk_size: int = 50000  # Chunk size for processing

@dataclass
class ProcessingConfig:
    """Main configuration for large data processing."""
    raster: RasterConfig = RasterConfig()
    vector: VectorConfig = VectorConfig()
    temp_dir: Path = Path('src/data/interim')
    max_memory: str = '4GB'  # Maximum memory usage
    n_workers: int = -1  # Number of workers (-1 for all available cores)
    
    # File paths
    input_paths: Dict[str, Path] = None
    output_paths: Dict[str, Path] = None
    
    def __post_init__(self):
        """Initialize default paths if not provided."""
        if self.input_paths is None:
            self.input_paths = {
                'ndvi': Path('src/data/raw/ndvi'),
                'parcels': Path('src/data/raw/gis'),
                'fia': Path('src/data/raw')
            }
        
        if self.output_paths is None:
            self.output_paths = {
                'processed_ndvi': Path('src/data/processed/ndvi'),
                'processed_parcels': Path('src/data/processed/parcels'),
                'results': Path('results')
            }
        
        # Ensure directories exist
        for path in self.output_paths.values():
            path.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True) 