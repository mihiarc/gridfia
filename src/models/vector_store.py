"""PostGIS integration for vector data storage."""
from typing import Optional, Union, List
import os
from pathlib import Path
import geopandas as gpd
from sqlalchemy import create_engine
from geoalchemy2 import Geometry
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    """Handles PostGIS operations for vector data."""
    
    def __init__(self, connection_string: Optional[str] = None):
        """Initialize PostGIS connection.
        
        Args:
            connection_string: Database connection string. If None, uses DATABASE_URL env var.
        """
        self.connection_string = connection_string or os.getenv('DATABASE_URL')
        if not self.connection_string:
            raise ValueError("Database connection string not provided")
        
        self.engine = create_engine(self.connection_string)
    
    def store_geodataframe(
        self,
        gdf: gpd.GeoDataFrame,
        table_name: str,
        if_exists: str = 'fail',
        schema: Optional[str] = None
    ) -> None:
        """Store GeoDataFrame in PostGIS.
        
        Args:
            gdf: GeoDataFrame to store
            table_name: Name of the table
            if_exists: How to behave if table exists ('fail', 'replace', 'append')
            schema: Database schema name
        """
        try:
            # Ensure SRID is set
            if gdf.crs is None:
                raise ValueError("GeoDataFrame must have a CRS")
            
            # Convert to PostGIS format
            gdf.to_postgis(
                table_name,
                self.engine,
                if_exists=if_exists,
                schema=schema,
                index=True,
                dtype={'geometry': Geometry(geometry_type='GEOMETRY', srid=gdf.crs.to_epsg())}
            )
            logger.info(f"Successfully stored {table_name} in PostGIS")
            
        except Exception as e:
            logger.error(f"Error storing {table_name}: {e}")
            raise
    
    def load_geodataframe(
        self,
        table_name: str,
        schema: Optional[str] = None,
        bbox: Optional[List[float]] = None,
        where: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """Load data from PostGIS as GeoDataFrame.
        
        Args:
            table_name: Name of the table
            schema: Database schema name
            bbox: Bounding box [minx, miny, maxx, maxy]
            where: SQL WHERE clause
            
        Returns:
            GeoDataFrame with query results
        """
        try:
            # Build query
            query = f"SELECT * FROM {schema+'.' if schema else ''}{table_name}"
            if bbox:
                bbox_wkt = f"ST_MakeEnvelope({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})"
                where_clause = f"ST_Intersects(geometry, {bbox_wkt})"
                if where:
                    where = f"{where} AND {where_clause}"
                else:
                    where = where_clause
            
            if where:
                query += f" WHERE {where}"
            
            # Load data
            gdf = gpd.read_postgis(
                query,
                self.engine,
                geom_col='geometry'
            )
            
            logger.info(f"Successfully loaded {table_name} from PostGIS")
            return gdf
            
        except Exception as e:
            logger.error(f"Error loading {table_name}: {e}")
            raise
    
    def create_spatial_index(
        self,
        table_name: str,
        schema: Optional[str] = None
    ) -> None:
        """Create spatial index on geometry column.
        
        Args:
            table_name: Name of the table
            schema: Database schema name
        """
        try:
            full_table = f"{schema+'.' if schema else ''}{table_name}"
            with self.engine.connect() as conn:
                conn.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{table_name}_geometry "
                    f"ON {full_table} USING GIST (geometry)"
                )
            logger.info(f"Created spatial index on {full_table}")
            
        except Exception as e:
            logger.error(f"Error creating spatial index on {table_name}: {e}")
            raise
    
    def execute_spatial_query(
        self,
        query: str,
        params: Optional[dict] = None
    ) -> gpd.GeoDataFrame:
        """Execute custom spatial query.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            GeoDataFrame with query results
        """
        try:
            return gpd.read_postgis(
                query,
                self.engine,
                params=params,
                geom_col='geometry'
            )
        except Exception as e:
            logger.error(f"Error executing spatial query: {e}")
            raise 