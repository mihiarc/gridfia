#!/bin/bash
set -e

echo "Setting up PostGIS with NC State Plane projection..."

# Wait for PostGIS to be ready
until docker-compose exec -T postgis pg_isready; do
    echo "Waiting for PostGIS..."
    sleep 2
done

# Run the setup script
echo "Running SRS setup script..."
docker-compose exec -T postgis psql -U postgres -d heirs_property -f /docker-entrypoint-initdb.d/01_setup_srs.sql

# Verify the setup
echo "Verifying NC State Plane projection..."
docker-compose exec -T postgis psql -U postgres -d heirs_property -c "SELECT srtext FROM spatial_ref_sys WHERE srid = 2264;"

echo "PostGIS setup complete!" 