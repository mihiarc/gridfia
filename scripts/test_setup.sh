#!/bin/bash
set -e

echo "Testing environment setup..."

# Create necessary directories
mkdir -p logs/postgis
mkdir -p data/{raw,processed,interim}

# Start containers
echo "Starting containers..."
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 10

# Run PostGIS setup
echo "Setting up PostGIS..."
./scripts/setup_postgis.sh

# Run tests
echo "Running tests..."
pytest tests/ -v

echo "Setup testing complete!" 