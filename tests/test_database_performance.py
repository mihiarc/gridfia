"""Database performance tests for the heirs-property project."""

import time
import pytest
import numpy as np
from typing import List, Tuple
from shapely.geometry import Point, Polygon

@pytest.fixture
def sample_points(db_cursor) -> List[Tuple[float, float]]:
    """Create sample points for testing."""
    # Create a test table for points
    db_cursor.execute("""
        CREATE TABLE IF NOT EXISTS test_points (
            id SERIAL PRIMARY KEY,
            geom GEOMETRY(Point, 4326)
        );
    """)
    
    # Generate random points in North Carolina bounds
    points = [
        (np.random.uniform(-84.3, -75.5), np.random.uniform(33.8, 36.6))
        for _ in range(1000)
    ]
    
    # Insert points
    for lon, lat in points:
        db_cursor.execute("""
            INSERT INTO test_points (geom)
            VALUES (ST_SetSRID(ST_MakePoint(%s, %s), 4326));
        """, (lon, lat))
    
    return points

@pytest.fixture
def sample_polygons(db_cursor) -> List[Polygon]:
    """Create sample polygons for testing."""
    # Create a test table for polygons
    db_cursor.execute("""
        CREATE TABLE IF NOT EXISTS test_polygons (
            id SERIAL PRIMARY KEY,
            geom GEOMETRY(Polygon, 4326)
        );
    """)
    
    # Generate random polygons
    polygons = []
    for _ in range(100):
        center_lon = np.random.uniform(-84.3, -75.5)
        center_lat = np.random.uniform(33.8, 36.6)
        
        # Create a random polygon around the center
        points = []
        for angle in range(0, 360, 60):
            radius = np.random.uniform(0.1, 0.5)
            lon = center_lon + radius * np.cos(np.radians(angle))
            lat = center_lat + radius * np.sin(np.radians(angle))
            points.append((lon, lat))
        points.append(points[0])  # Close the polygon
        
        # Insert polygon
        db_cursor.execute("""
            INSERT INTO test_polygons (geom)
            VALUES (ST_SetSRID(ST_MakePolygon(ST_MakeLine(ARRAY[%s])), 4326));
        """, ([f"POINT({lon} {lat})" for lon, lat in points],))
        
        polygons.append(points)
    
    return polygons

@pytest.mark.benchmark
def test_point_queries(db_cursor, sample_points):
    """Test point query performance."""
    start_time = time.time()
    
    # Test point in polygon query
    db_cursor.execute("""
        SELECT COUNT(*)
        FROM test_points p
        JOIN test_polygons poly
        ON ST_Contains(poly.geom, p.geom);
    """)
    count = db_cursor.fetchone()[0]
    
    duration = time.time() - start_time
    assert duration < 2.0, f"Point query too slow: {duration:.2f}s"
    print(f"Point query completed in {duration:.2f}s, found {count} intersections")

@pytest.mark.benchmark
def test_spatial_joins(db_cursor):
    """Test spatial join performance."""
    start_time = time.time()
    
    # Test spatial join with transformation
    db_cursor.execute("""
        SELECT COUNT(*)
        FROM test_points p
        JOIN test_polygons poly
        ON ST_Contains(
            ST_Transform(poly.geom, 2264),
            ST_Transform(p.geom, 2264)
        );
    """)
    count = db_cursor.fetchone()[0]
    
    duration = time.time() - start_time
    assert duration < 5.0, f"Spatial join too slow: {duration:.2f}s"
    print(f"Spatial join completed in {duration:.2f}s, found {count} matches")

@pytest.mark.benchmark
def test_area_calculations(db_cursor):
    """Test area calculation performance."""
    start_time = time.time()
    
    # Calculate areas in NC State Plane
    db_cursor.execute("""
        SELECT COUNT(*)
        FROM (
            SELECT ST_Area(ST_Transform(geom, 2264)) as area
            FROM test_polygons
            WHERE ST_Area(ST_Transform(geom, 2264)) > 1000
        ) as large_areas;
    """)
    count = db_cursor.fetchone()[0]
    
    duration = time.time() - start_time
    assert duration < 1.0, f"Area calculation too slow: {duration:.2f}s"
    print(f"Area calculation completed in {duration:.2f}s, found {count} large areas")

@pytest.mark.benchmark
def test_buffer_operations(db_cursor):
    """Test buffer operation performance."""
    start_time = time.time()
    
    # Create buffers and check intersections
    db_cursor.execute("""
        SELECT COUNT(*)
        FROM test_points p1
        JOIN test_points p2
        ON ST_Intersects(
            ST_Buffer(ST_Transform(p1.geom, 2264), 1000),
            ST_Transform(p2.geom, 2264)
        )
        WHERE p1.id != p2.id;
    """)
    count = db_cursor.fetchone()[0]
    
    duration = time.time() - start_time
    assert duration < 10.0, f"Buffer operation too slow: {duration:.2f}s"
    print(f"Buffer operation completed in {duration:.2f}s, found {count} intersections")

@pytest.mark.benchmark
def test_index_performance(db_cursor):
    """Test spatial index performance."""
    # Test query without index
    db_cursor.execute("DROP INDEX IF EXISTS idx_test_points_geom;")
    
    start_time = time.time()
    db_cursor.execute("""
        SELECT COUNT(*)
        FROM test_points
        WHERE ST_Within(
            geom,
            ST_MakeEnvelope(-80, 35, -79, 36, 4326)
        );
    """)
    no_index_time = time.time() - start_time
    
    # Create index and test again
    db_cursor.execute("""
        CREATE INDEX idx_test_points_geom
        ON test_points USING GIST (geom);
    """)
    
    start_time = time.time()
    db_cursor.execute("""
        SELECT COUNT(*)
        FROM test_points
        WHERE ST_Within(
            geom,
            ST_MakeEnvelope(-80, 35, -79, 36, 4326)
        );
    """)
    with_index_time = time.time() - start_time
    
    # Index should provide significant improvement
    assert with_index_time < no_index_time / 2, \
        f"Index not providing sufficient improvement: {with_index_time:.2f}s vs {no_index_time:.2f}s"
    print(f"Query time improved from {no_index_time:.2f}s to {with_index_time:.2f}s with index")

def test_cleanup(db_cursor):
    """Clean up test tables."""
    db_cursor.execute("DROP TABLE IF EXISTS test_points;")
    db_cursor.execute("DROP TABLE IF EXISTS test_polygons;") 