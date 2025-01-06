"""
Parcel data processor for converting and importing parcel data into PostGIS.
"""
from pathlib import Path
import logging
from typing import Optional, Dict, Tuple, Callable
import os
import time

import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine.base import Engine

from .data_validator import DataValidator
from .chunked_processor import ChunkedProcessor, ProcessingStats
from .transaction_manager import TransactionManager
from .pipeline_monitor import PipelineMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParcelProcessor:
    """Handles the processing and import of parcel data into PostGIS."""
    
    def __init__(
        self,
        db_connection: str,
        schema: str = "public",
        srid: int = 2264,  # NC State Plane
        required_fields: Optional[list] = None,
        chunk_size: int = 10000,
        max_workers: Optional[int] = None,
        memory_limit_mb: int = 1000,
        checkpoint_dir: Optional[Path] = None,
        metrics_dir: Optional[Path] = None,
        validator: Optional[DataValidator] = None,
        chunked_processor: Optional[ChunkedProcessor] = None,
        transaction_manager: Optional[TransactionManager] = None,
        monitor: Optional[PipelineMonitor] = None
    ):
        """
        Initialize the parcel processor.
        
        Args:
            db_connection: PostgreSQL connection string
            schema: Target schema name
            srid: Spatial reference system ID (default: NC State Plane)
            required_fields: List of required fields (optional)
            chunk_size: Number of records per chunk
            max_workers: Maximum number of parallel workers
            memory_limit_mb: Maximum memory usage in MB
            checkpoint_dir: Directory for storing transaction checkpoints
            metrics_dir: Directory for storing monitoring metrics
            validator: Optional DataValidator instance
            chunked_processor: Optional ChunkedProcessor instance
            transaction_manager: Optional TransactionManager instance
            monitor: Optional PipelineMonitor instance
        """
        self.engine = create_engine(db_connection)
        self.schema = schema
        self.srid = srid
        self.required_fields = required_fields or ['geometry']
        
        # Use provided components or create new ones
        self.validator = validator or DataValidator(
            required_fields=self.required_fields,
            geometry_type='Polygon',
            srid=self.srid
        )
        self.chunked_processor = chunked_processor or ChunkedProcessor(
            chunk_size=chunk_size,
            max_workers=max_workers,
            memory_limit_mb=memory_limit_mb
        )
        self.transaction_manager = transaction_manager or TransactionManager(
            db_connection=db_connection,
            checkpoint_dir=checkpoint_dir
        )
        
        # Use provided monitor or create new one if metrics directory is provided
        self.monitor = monitor
        if not monitor and metrics_dir:
            self.monitor = PipelineMonitor(
                metrics_dir=metrics_dir,
                resource_interval=30,  # Monitor every 30 seconds
                alert_cpu_threshold=80.0,
                alert_memory_threshold=80.0,
                alert_disk_threshold=80.0
            )
        
    def validate_data(
        self,
        gdf: gpd.GeoDataFrame,
        dataset_name: str
    ) -> Tuple[bool, Dict]:
        """
        Validate the data before processing.
        
        Args:
            gdf: GeoDataFrame to validate
            dataset_name: Name of the dataset
            
        Returns:
            Tuple of (success, validation_report)
        """
        start_time = time.time()
        success, report = self.validator.validate_geodataframe(gdf, dataset_name)
        
        if self.monitor:
            validation_time = time.time() - start_time
            self.monitor.update_processing_metrics(
                dataset_name=dataset_name,
                validation_errors=len(report.get("issues", [])),
                processing_time=validation_time
            )
        
        return success, report
    
    def _prepare_for_import(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Prepare a chunk for import to PostGIS.
        
        Args:
            gdf: GeoDataFrame chunk to prepare
            
        Returns:
            Prepared GeoDataFrame
        """
        # Ensure geometry is in correct SRID
        if gdf.crs is None or gdf.crs.to_epsg() != self.srid:
            gdf = gdf.to_crs(epsg=self.srid)
        return gdf
        
    def process_parquet(
        self,
        parquet_path: Path,
        table_name: str,
        geometry_column: str = "geometry",
        validation_report_path: Optional[Path] = None,
        stats_path: Optional[Path] = None
    ) -> bool:
        """
        Process parquet file and import to PostGIS.
        
        Args:
            parquet_path: Path to parquet file
            table_name: Target table name
            geometry_column: Name of geometry column
            validation_report_path: Path to save validation report
            stats_path: Path to save processing stats
            
        Returns:
            bool: True if successful
        """
        try:
            # Start monitoring if available
            if self.monitor:
                self.monitor.start_monitoring()
            
            logger.info(f"Reading parquet file: {parquet_path}")
            gdf = gpd.read_parquet(parquet_path)
            
            # Initialize processing metrics
            if self.monitor:
                self.monitor.start_processing_metrics(
                    dataset_name=table_name,
                    total_records=len(gdf)
                )
            
            # Validate data
            success, report = self.validate_data(gdf, table_name)
            if validation_report_path:
                validation_report_path.parent.mkdir(parents=True, exist_ok=True)
                with open(validation_report_path, 'w') as f:
                    import json
                    json.dump(report, f, indent=2)
            
            if not success:
                logger.error(f"Validation failed for {table_name}")
                logger.error(f"Validation report: {report}")
                return False
            
            # Process in chunks
            logger.info("Processing data in chunks...")
            processed_gdf, stats = self.chunked_processor.process_geodataframe(
                gdf,
                self._prepare_for_import
            )
            
            if stats_path:
                stats_path.parent.mkdir(parents=True, exist_ok=True)
                with open(stats_path, 'w') as f:
                    json.dump(stats.to_dict(), f, indent=2)
            
            if processed_gdf is None:
                logger.error("Failed to process data")
                return False
            
            # Update processing metrics
            if self.monitor:
                self.monitor.update_processing_metrics(
                    dataset_name=table_name,
                    processed_records=stats.processed_records,
                    failed_records=stats.total_records - stats.processed_records,
                    processing_errors=len(stats.failed_chunks),
                    memory_mb=max(stats.memory_usage.values())
                )
            
            # Import to PostGIS with transaction management
            logger.info(f"Importing to PostGIS table: {table_name}")
            with self.transaction_manager.transaction(table_name, self.schema) as engine:
                processed_gdf.to_postgis(
                    table_name,
                    engine,
                    schema=self.schema,
                    if_exists="replace",
                    index=True,
                    dtype={"geometry": f"geometry(Polygon, {self.srid})"}
                )
                
                # Create spatial index
                with engine.connect() as conn:
                    conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{table_name}_geom 
                    ON {self.schema}.{table_name} USING GIST ({geometry_column});
                    """)
                    conn.commit()
            
            logger.info("Import completed successfully")
            
            # Complete processing metrics
            if self.monitor:
                self.monitor.complete_processing_metrics(table_name)
                self.monitor.save_metrics()
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing parcel data: {str(e)}")
            return False
        
        finally:
            # Stop monitoring
            if self.monitor:
                self.monitor.stop_monitoring()
            
    def process_gdb(
        self,
        gdb_path: Path,
        layer_name: str,
        table_name: str,
        geometry_column: str = "geometry",
        validation_report_path: Optional[Path] = None,
        stats_path: Optional[Path] = None
    ) -> bool:
        """
        Process GDB file and import to PostGIS.
        
        Args:
            gdb_path: Path to GDB file
            layer_name: Name of the layer in GDB
            table_name: Target table name
            geometry_column: Name of geometry column
            validation_report_path: Path to save validation report
            stats_path: Path to save processing stats
            
        Returns:
            bool: True if successful
        """
        try:
            # Start monitoring if available
            if self.monitor:
                self.monitor.start_monitoring()
            
            logger.info(f"Reading GDB layer: {layer_name}")
            gdf = gpd.read_file(gdb_path, layer=layer_name)
            
            # Initialize processing metrics
            if self.monitor:
                self.monitor.start_processing_metrics(
                    dataset_name=table_name,
                    total_records=len(gdf)
                )
            
            # Validate data
            success, report = self.validate_data(gdf, table_name)
            if validation_report_path:
                validation_report_path.parent.mkdir(parents=True, exist_ok=True)
                with open(validation_report_path, 'w') as f:
                    import json
                    json.dump(report, f, indent=2)
            
            if not success:
                logger.error(f"Validation failed for {table_name}")
                logger.error(f"Validation report: {report}")
                return False
            
            # Process in chunks
            logger.info("Processing data in chunks...")
            processed_gdf, stats = self.chunked_processor.process_geodataframe(
                gdf,
                self._prepare_for_import
            )
            
            if stats_path:
                stats_path.parent.mkdir(parents=True, exist_ok=True)
                with open(stats_path, 'w') as f:
                    json.dump(stats.to_dict(), f, indent=2)
            
            if processed_gdf is None:
                logger.error("Failed to process data")
                return False
            
            # Update processing metrics
            if self.monitor:
                self.monitor.update_processing_metrics(
                    dataset_name=table_name,
                    processed_records=stats.processed_records,
                    failed_records=stats.total_records - stats.processed_records,
                    processing_errors=len(stats.failed_chunks),
                    memory_mb=max(stats.memory_usage.values())
                )
            
            # Import to PostGIS with transaction management
            logger.info(f"Importing to PostGIS table: {table_name}")
            with self.transaction_manager.transaction(table_name, self.schema) as engine:
                processed_gdf.to_postgis(
                    table_name,
                    engine,
                    schema=self.schema,
                    if_exists="replace",
                    index=True,
                    dtype={"geometry": f"geometry(Polygon, {self.srid})"}
                )
                
                # Create spatial index
                with engine.connect() as conn:
                    conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{table_name}_geom 
                    ON {self.schema}.{table_name} USING GIST ({geometry_column});
                    """)
                    conn.commit()
            
            logger.info("Import completed successfully")
            
            # Complete processing metrics
            if self.monitor:
                self.monitor.complete_processing_metrics(table_name)
                self.monitor.save_metrics()
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing GDB data: {str(e)}")
            return False
        
        finally:
            # Stop monitoring
            if self.monitor:
                self.monitor.stop_monitoring() 