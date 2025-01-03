-- Install required extensions
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;
CREATE EXTENSION IF NOT EXISTS postgis_tiger_geocoder CASCADE;

-- Enable additional PostGIS extensions for advanced functionality
CREATE EXTENSION IF NOT EXISTS postgis_sfcgal;    -- For advanced 3D operations
CREATE EXTENSION IF NOT EXISTS postgis_tiger_geocoder;  -- For geocoding support
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;      -- Required for geocoding
CREATE EXTENSION IF NOT EXISTS address_standardizer; -- For address normalization

-- Configure spatial reference systems
-- Add common NC State Plane coordinate systems
INSERT INTO spatial_ref_sys (srid, auth_name, auth_srid, proj4text, srtext)
SELECT 
    2264, 'EPSG', 2264,
    '+proj=lcc +lat_1=36.16666666666666 +lat_2=34.33333333333334 +lat_0=33.75 +lon_0=-79 +x_0=609601.22 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs',
    'PROJCS["NAD83 / North Carolina (ftUS)",GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4269"]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["standard_parallel_1",36.16666666666666],PARAMETER["standard_parallel_2",34.33333333333334],PARAMETER["latitude_of_origin",33.75],PARAMETER["central_meridian",-79],PARAMETER["false_easting",2000000],PARAMETER["false_northing",0],UNIT["US survey foot",0.3048006096012192,AUTHORITY["EPSG","9003"]],AXIS["X",EAST],AXIS["Y",NORTH],AUTHORITY["EPSG","2264"]]'
WHERE NOT EXISTS (
    SELECT 1 FROM spatial_ref_sys WHERE srid = 2264
);

-- Create functions for coordinate transformations
CREATE OR REPLACE FUNCTION vector.transform_to_nc_stateplanes(
    geom geometry
)
RETURNS geometry AS $$
BEGIN
    -- Transform from WGS84 (EPSG:4326) to NC State Plane (EPSG:2264)
    RETURN ST_Transform(geom, 2264);
END;
$$ LANGUAGE plpgsql;

-- Optimize PostGIS settings
ALTER DATABASE heirs_property SET postgis.gdal_enabled_drivers TO 'ENABLE_ALL';
ALTER DATABASE heirs_property SET postgis.enable_outdb_rasters TO True;

-- Create spatial functions for area calculations in acres
CREATE OR REPLACE FUNCTION vector.calculate_area_acres(
    geom geometry
)
RETURNS DOUBLE PRECISION AS $$
BEGIN
    -- Convert square meters to acres
    RETURN ST_Area(geom::geography) * 0.000247105;
END;
$$ LANGUAGE plpgsql;

-- Add area calculation triggers
CREATE OR REPLACE FUNCTION vector.update_area_trigger()
RETURNS TRIGGER AS $$
BEGIN
    NEW.area_sqm := ST_Area(NEW.geometry::geography);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_parcel_area
    BEFORE INSERT OR UPDATE OF geometry
    ON vector.parcels
    FOR EACH ROW
    EXECUTE FUNCTION vector.update_area_trigger();

CREATE TRIGGER update_hp_area
    BEFORE INSERT OR UPDATE OF geometry
    ON vector.heirs_properties
    FOR EACH ROW
    EXECUTE FUNCTION vector.update_area_trigger();

-- Create spatial validation functions
CREATE OR REPLACE FUNCTION vector.validate_geometry(
    geom geometry
)
RETURNS TABLE (
    is_valid boolean,
    validation_error text
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ST_IsValid(geom),
        CASE 
            WHEN NOT ST_IsValid(geom) THEN ST_IsValidReason(geom)
            ELSE 'Valid geometry'
        END;
END;
$$ LANGUAGE plpgsql;

-- Create spatial index maintenance function
CREATE OR REPLACE FUNCTION vector.maintain_spatial_indices()
RETURNS void AS $$
BEGIN
    -- Rebuild spatial indices
    REINDEX INDEX vector.idx_parcels_geometry;
    REINDEX INDEX vector.idx_heirs_geometry;
    
    -- Analyze tables for query optimization
    ANALYZE vector.parcels;
    ANALYZE vector.heirs_properties;
END;
$$ LANGUAGE plpgsql; 