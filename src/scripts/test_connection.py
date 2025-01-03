"""
Test script to verify database connection and PostGIS functionality.
"""
import os
from sqlalchemy import create_engine, text
import geopandas as gpd
import pandas as pd

def test_db_connection():
    """Test connection to PostGIS database."""
    # Create connection string
    db_params = {
        'host': os.getenv('DB_HOST', 'postgis'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME', 'heirs_property'),
        'user': os.getenv('DB_USER', 'heirs_user'),
        'password': os.getenv('DB_PASSWORD', 'dev_password_123')
    }
    
    conn_string = f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}"
    
    try:
        # Create engine
        engine = create_engine(conn_string)
        
        # Test connection
        with engine.connect() as conn:
            # Test basic connection
            result = conn.execute(text("SELECT 1"))
            print("Basic connection test: Success")
            
            # Test PostGIS installation
            result = conn.execute(text("SELECT PostGIS_Version()"))
            version = result.scalar()
            print(f"PostGIS version: {version}")
            
            # Test spatial reference system
            result = conn.execute(text("SELECT srtext FROM spatial_ref_sys WHERE srid = 2264"))
            srs = result.scalar()
            print(f"NC State Plane definition found: {'Success' if srs else 'Failed'}")
            
        return True
    except Exception as e:
        print(f"Connection test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing database connection...")
    success = test_db_connection()
    print(f"Overall test {'succeeded' if success else 'failed'}") 