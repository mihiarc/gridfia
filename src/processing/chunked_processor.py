"""
Chunked processing module for large datasets.
"""
import time
from typing import Dict, Any, Callable, Optional
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import pandas as pd
import geopandas as gpd
import psutil

@dataclass
class ProcessingStats:
    """Statistics for chunked processing."""
    start_time: str = field(default_factory=lambda: time.time())
    end_time: Optional[str] = None
    total_chunks: int = 0
    processed_chunks: int = 0
    failed_chunks: list = field(default_factory=list)
    total_records: int = 0
    processed_records: int = 0
    memory_usage: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_chunks": self.total_chunks,
            "processed_chunks": self.processed_chunks,
            "failed_chunks": self.failed_chunks,
            "total_records": self.total_records,
            "processed_records": self.processed_records,
            "memory_usage": self.memory_usage,
            "success_rate": (
                self.processed_chunks / self.total_chunks 
                if self.total_chunks > 0 else 0
            )
        }

class ChunkedProcessor:
    """Processes large datasets in chunks."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        max_workers: int = 4,
        memory_limit_mb: float = 1000.0
    ):
        """Initialize processor with configuration.
        
        Args:
            chunk_size: Number of records per chunk
            max_workers: Maximum number of parallel workers
            memory_limit_mb: Memory limit in megabytes
        """
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.memory_limit_mb = memory_limit_mb
    
    def _monitor_memory(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _process_chunk(
        self,
        chunk: gpd.GeoDataFrame,
        process_func: Callable,
        chunk_id: int
    ) -> tuple[Optional[gpd.GeoDataFrame], Dict[str, Any]]:
        """Process a single chunk.
        
        Args:
            chunk: Data chunk to process
            process_func: Processing function to apply
            chunk_id: Identifier for the chunk
            
        Returns:
            Tuple of (processed_chunk, chunk_stats)
        """
        start_time = time.time()
        start_memory = self._monitor_memory()
        
        try:
            print(f"Processing chunk {chunk_id} with shape {chunk.shape}")
            result = process_func(chunk)
            print(f"Chunk {chunk_id} processed, result type: {type(result)}")
            
            if not isinstance(result, gpd.GeoDataFrame):
                raise ValueError("Process function must return a GeoDataFrame")
            
            # Only modify if the geometry column or CRS changed
            if result.geometry.name != chunk.geometry.name:
                print(f"Chunk {chunk_id}: Geometry column changed from {chunk.geometry.name} to {result.geometry.name}")
                result = result.rename_geometry(chunk.geometry.name)
            if result.crs != chunk.crs and chunk.crs is not None:
                print(f"Chunk {chunk_id}: CRS changed from {chunk.crs} to {result.crs}")
                result = result.set_crs(chunk.crs)
                
            success = True
            print(f"Chunk {chunk_id} successfully processed")
        except Exception as e:
            print(f"Error processing chunk {chunk_id}: {str(e)}")
            result = None
            success = False
        
        end_time = time.time()
        end_memory = self._monitor_memory()
        
        stats = {
            "chunk_id": chunk_id,
            "success": success,
            "processing_time": end_time - start_time,
            "memory_used_mb": end_memory - start_memory,
            "records": len(chunk)
        }
        
        return result, stats
    
    def process_geodataframe(
        self,
        gdf: gpd.GeoDataFrame,
        process_func: Callable,
        progress_bar: bool = False
    ) -> tuple[Optional[gpd.GeoDataFrame], ProcessingStats]:
        """Process a GeoDataFrame in chunks.
        
        Args:
            gdf: GeoDataFrame to process
            process_func: Function to apply to each chunk
            progress_bar: Whether to show progress bar
            
        Returns:
            Tuple of (processed_data, processing_stats)
        """
        if not isinstance(gdf, gpd.GeoDataFrame):
            raise ValueError("Input must be a GeoDataFrame")
            
        print(f"Starting processing of GeoDataFrame with shape {gdf.shape}")
        print(f"Input GeoDataFrame CRS: {gdf.crs}")
        print(f"Input geometry column: {gdf.geometry.name}")
            
        stats = ProcessingStats()
        stats.total_records = len(gdf)
        
        # Store original properties
        geometry_column = gdf.geometry.name
        original_crs = gdf.crs
        
        # Split into chunks
        chunks = [
            gdf.iloc[i:i + self.chunk_size].copy()  # Create a copy to avoid view issues
            for i in range(0, len(gdf), self.chunk_size)
        ]
        stats.total_chunks = len(chunks)
        print(f"Split into {len(chunks)} chunks")
        
        processed_chunks = []
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(
                    self._process_chunk,
                    chunk,
                    process_func,
                    i
                )
                for i, chunk in enumerate(chunks)
            ]
            
            for future in futures:
                try:
                    result, chunk_stats = future.result()
                    if chunk_stats["success"] and result is not None:
                        # Ensure chunk has correct CRS
                        if result.crs != original_crs:
                            print(f"Correcting CRS from {result.crs} to {original_crs}")
                            result = result.set_crs(original_crs)
                        processed_chunks.append(result)
                        stats.processed_chunks += 1
                        stats.processed_records += chunk_stats["records"]
                    else:
                        stats.failed_chunks.append(chunk_stats["chunk_id"])
                    
                    stats.memory_usage[f"chunk_{chunk_stats['chunk_id']}"] = (
                        chunk_stats["memory_used_mb"]
                    )
                except Exception as e:
                    print(f"Error in future: {str(e)}")
                    stats.failed_chunks.append(len(processed_chunks))
        
        stats.end_time = time.time()
        
        if not processed_chunks:
            print("No chunks were processed successfully")
            return None, stats
        
        try:
            print(f"Concatenating {len(processed_chunks)} processed chunks")
            # Ensure we preserve the GeoDataFrame properties
            result = pd.concat(processed_chunks, ignore_index=True)
            result = gpd.GeoDataFrame(
                result,
                geometry=geometry_column,
                crs=original_crs
            )
            print(f"Final result shape: {result.shape}")
            return result, stats
        except Exception as e:
            print(f"Error in final concatenation: {str(e)}")
            return None, stats
    
    def process_parquet(
        self,
        file_path: str,
        process_func: Callable,
        progress_bar: bool = False
    ) -> tuple[Optional[gpd.GeoDataFrame], ProcessingStats]:
        """Process a parquet file in chunks.
        
        Args:
            file_path: Path to parquet file
            process_func: Function to apply to each chunk
            progress_bar: Whether to show progress bar
            
        Returns:
            Tuple of (processed_data, processing_stats)
        """
        try:
            # Try reading as GeoParquet first
            gdf = gpd.read_parquet(file_path)
        except ValueError:
            # If not a GeoParquet, read as regular parquet and convert geometry
            df = pd.read_parquet(file_path)
            if 'geometry' in df.columns:
                gdf = gpd.GeoDataFrame(
                    df.drop(columns=['geometry']),
                    geometry=gpd.GeoSeries.from_wkt(df['geometry']),
                    crs="EPSG:4326"  # Default to WGS84
                )
            else:
                raise ValueError("No geometry column found in parquet file")
        
        return self.process_geodataframe(
            gdf,
            process_func,
            progress_bar
        ) 