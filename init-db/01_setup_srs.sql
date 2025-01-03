-- Verify PostGIS installation
SELECT PostGIS_Version();

-- Check if NC State Plane (NAD83) exists
SELECT * FROM spatial_ref_sys WHERE srid = 2264;

-- If not exists, add NC State Plane (NAD83) definition
INSERT INTO spatial_ref_sys (srid, auth_name, auth_srid, proj4text, srtext) 
SELECT 
    2264,
    'EPSG',
    2264,
    '+proj=lcc +lat_1=34.33333333333334 +lat_2=36.16666666666666 +lat_0=33.75 +lon_0=-79 +x_0=609601.22 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs',
    'PROJCS["NAD83 / North Carolina (ftUS)",GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4269"]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["standard_parallel_1",34.33333333333334],PARAMETER["standard_parallel_2",36.16666666666666],PARAMETER["latitude_of_origin",33.75],PARAMETER["central_meridian",-79],PARAMETER["false_easting",2000000],PARAMETER["false_northing",0],UNIT["US survey foot",0.3048006096012192,AUTHORITY["EPSG","9003"]],AXIS["X",EAST],AXIS["Y",NORTH],AUTHORITY["EPSG","2264"]]'
WHERE NOT EXISTS (
    SELECT 1 FROM spatial_ref_sys WHERE srid = 2264
);

-- Create function to transform coordinates between WGS84 and NC State Plane
CREATE OR REPLACE FUNCTION public.transform_to_nc_state_plane(
    geom geometry
) RETURNS geometry AS $$
BEGIN
    RETURN ST_Transform(geom, 2264);
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT;

CREATE OR REPLACE FUNCTION public.transform_from_nc_state_plane(
    geom geometry
) RETURNS geometry AS $$
BEGIN
    RETURN ST_Transform(geom, 4326);
END;
$$ LANGUAGE plpgsql IMMUTABLE STRICT; 