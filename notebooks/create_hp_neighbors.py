import geopandas as gpd
from shapely.prepared import prep

# Read the datasets
parcels_nc = gpd.read_parquet('/Users/mihiarc/Work/data/HeirsParcels/nc-parcels.parquet')
parcels_hp = gpd.read_parquet('/Users/mihiarc/Work/data/HeirsParcels/nc-hp_v2.parquet')

# Ensure both GeoDataFrames use the same projected coordinate system
parcels_nc = parcels_nc.to_crs(epsg=32617)
parcels_hp = parcels_hp.to_crs(epsg=32617)

# Combine all geometries in parcels_hp into a single geometry
parcels_hp_union = parcels_hp.union_all()

# Create a buffer of 1 mile (1609.34 meters) around the combined geometry
buffer_distance = 1609.34  # 1 mile in meters
parcels_hp_buffer = parcels_hp_union.buffer(buffer_distance)

# Prepare the buffer geometry for faster spatial operations
prepared_buffer = prep(parcels_hp_buffer)

# Use spatial indexing to find candidate parcels within the buffer's bounding box
candidates_index = list(parcels_nc.sindex.intersection(parcels_hp_buffer.bounds))
candidates = parcels_nc.iloc[candidates_index]

# Filter candidates to those that intersect with the buffer
parcels_within_1_mile = candidates[candidates['geometry'].apply(prepared_buffer.intersects)]

# Save the result to a Parquet file
parcels_within_1_mile.to_parquet('/Users/mihiarc/Work/data/HeirsParcels/parcels_within_1_mile.parquet')