-- Enable PostGIS extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_raster;
CREATE EXTENSION IF NOT EXISTS postgis_topology;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS raster;
CREATE SCHEMA IF NOT EXISTS analysis;

-- Create tables for vector data
CREATE TABLE IF NOT EXISTS vector.parcels (
    id SERIAL PRIMARY KEY,
    parcel_id VARCHAR(50) UNIQUE NOT NULL,
    fips VARCHAR(5) NOT NULL,
    geometry geometry(MULTIPOLYGON, 4326),
    area_sqm DOUBLE PRECISION,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS vector.heirs_properties (
    id SERIAL PRIMARY KEY,
    hp_id VARCHAR(50) UNIQUE NOT NULL,
    parcel_id VARCHAR(50) REFERENCES vector.parcels(parcel_id),
    geometry geometry(MULTIPOLYGON, 4326),
    area_sqm DOUBLE PRECISION,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create tables for NDVI analysis
CREATE TABLE IF NOT EXISTS analysis.ndvi_stats (
    id SERIAL PRIMARY KEY,
    parcel_id VARCHAR(50) REFERENCES vector.parcels(parcel_id),
    capture_date DATE NOT NULL,
    mean_ndvi DOUBLE PRECISION,
    std_ndvi DOUBLE PRECISION,
    min_ndvi DOUBLE PRECISION,
    max_ndvi DOUBLE PRECISION,
    pixel_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indices
CREATE INDEX IF NOT EXISTS idx_parcels_geometry ON vector.parcels USING GIST (geometry);
CREATE INDEX IF NOT EXISTS idx_heirs_geometry ON vector.heirs_properties USING GIST (geometry);
CREATE INDEX IF NOT EXISTS idx_parcels_fips ON vector.parcels (fips);
CREATE INDEX IF NOT EXISTS idx_ndvi_date ON analysis.ndvi_stats (capture_date);
CREATE INDEX IF NOT EXISTS idx_ndvi_parcel ON analysis.ndvi_stats (parcel_id);

-- Create views
CREATE OR REPLACE VIEW analysis.parcel_ndvi_summary AS
SELECT 
    p.parcel_id,
    p.fips,
    hp.hp_id,
    p.area_sqm,
    n.capture_date,
    n.mean_ndvi,
    n.std_ndvi,
    ST_AsGeoJSON(p.geometry)::json as geometry
FROM 
    vector.parcels p
    LEFT JOIN vector.heirs_properties hp ON p.parcel_id = hp.parcel_id
    LEFT JOIN analysis.ndvi_stats n ON p.parcel_id = n.parcel_id;

-- Create functions for spatial analysis
CREATE OR REPLACE FUNCTION vector.get_neighboring_parcels(
    target_parcel_id VARCHAR,
    buffer_meters DOUBLE PRECISION DEFAULT 1609.34  -- 1 mile
)
RETURNS TABLE (
    parcel_id VARCHAR,
    distance_meters DOUBLE PRECISION,
    is_heir_property BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    WITH target AS (
        SELECT geometry FROM vector.parcels WHERE parcel_id = target_parcel_id
    )
    SELECT 
        p.parcel_id,
        ST_Distance(p.geometry::geography, t.geometry::geography) as distance_meters,
        CASE WHEN hp.hp_id IS NOT NULL THEN TRUE ELSE FALSE END as is_heir_property
    FROM 
        vector.parcels p
        CROSS JOIN target t
        LEFT JOIN vector.heirs_properties hp ON p.parcel_id = hp.parcel_id
    WHERE 
        p.parcel_id != target_parcel_id
        AND ST_DWithin(p.geometry::geography, t.geometry::geography, buffer_meters)
    ORDER BY 
        distance_meters;
END;
$$ LANGUAGE plpgsql;

-- Create function to calculate NDVI statistics for a parcel
CREATE OR REPLACE FUNCTION analysis.calculate_parcel_ndvi_stats(
    target_parcel_id VARCHAR,
    capture_date DATE
)
RETURNS TABLE (
    mean_ndvi DOUBLE PRECISION,
    std_ndvi DOUBLE PRECISION,
    min_ndvi DOUBLE PRECISION,
    max_ndvi DOUBLE PRECISION,
    pixel_count INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        AVG(n.mean_ndvi) as mean_ndvi,
        STDDEV(n.mean_ndvi) as std_ndvi,
        MIN(n.min_ndvi) as min_ndvi,
        MAX(n.max_ndvi) as max_ndvi,
        SUM(n.pixel_count) as pixel_count
    FROM 
        analysis.ndvi_stats n
    WHERE 
        n.parcel_id = target_parcel_id
        AND n.capture_date = calculate_parcel_ndvi_stats.capture_date;
END;
$$ LANGUAGE plpgsql; 