# Next Steps: Database Schema and Data Pipeline Implementation

## Overview
This document outlines the immediate next steps for implementing the database schema and data processing pipeline for the Heirs Property Analysis project.

## 1. Database Schema Implementation

### A. Core Tables
```sql
-- Parcels table
CREATE TABLE parcels (
    id SERIAL PRIMARY KEY,
    parcel_id VARCHAR(50) UNIQUE NOT NULL,
    geom GEOMETRY(POLYGON, 2264),
    area_acres DOUBLE PRECISION,
    county VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Heirs properties table
CREATE TABLE heirs_properties (
    id SERIAL PRIMARY KEY,
    hp_id VARCHAR(50) UNIQUE NOT NULL,
    parcel_id VARCHAR(50) REFERENCES parcels(parcel_id),
    geom GEOMETRY(POLYGON, 2264),
    area_acres DOUBLE PRECISION,
    verification_status VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- FIA plots table
CREATE TABLE fia_plots (
    id SERIAL PRIMARY KEY,
    plot_id VARCHAR(50) UNIQUE NOT NULL,
    geom GEOMETRY(POINT, 2264),
    measurement_year INTEGER,
    forest_type VARCHAR(100),
    stand_age INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- NDVI statistics table
CREATE TABLE ndvi_stats (
    id SERIAL PRIMARY KEY,
    parcel_id VARCHAR(50) REFERENCES parcels(parcel_id),
    capture_date DATE,
    mean_ndvi DOUBLE PRECISION,
    std_ndvi DOUBLE PRECISION,
    min_ndvi DOUBLE PRECISION,
    max_ndvi DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### B. Spatial Indices
```sql
-- Create spatial indices
CREATE INDEX parcels_geom_idx ON parcels USING GIST (geom);
CREATE INDEX heirs_properties_geom_idx ON heirs_properties USING GIST (geom);
CREATE INDEX fia_plots_geom_idx ON fia_plots USING GIST (geom);
```

### C. Views
```sql
-- Create views for common queries
CREATE VIEW heirs_property_summary AS
SELECT 
    hp.hp_id,
    hp.area_acres,
    p.county,
    COUNT(f.plot_id) as fia_plot_count,
    AVG(n.mean_ndvi) as avg_ndvi
FROM heirs_properties hp
JOIN parcels p ON hp.parcel_id = p.parcel_id
LEFT JOIN fia_plots f ON ST_DWithin(hp.geom, f.geom, 1609.34) -- 1 mile
LEFT JOIN ndvi_stats n ON hp.parcel_id = n.parcel_id
GROUP BY hp.hp_id, hp.area_acres, p.county;
```

## 2. Data Import Pipeline

### A. GDB Processing
```python
class GDBProcessor:
    """Process GDB files into Parquet format."""
    
    def process_gdb(self, gdb_path: Path, layer_name: str) -> Path:
        """
        Process a GDB file into Parquet format.
        
        Args:
            gdb_path: Path to the GDB file
            layer_name: Name of the layer to process
            
        Returns:
            Path to the output Parquet file
        """
        # Implementation steps:
        # 1. Read GDB layer
        # 2. Clean and standardize fields
        # 3. Convert to Parquet
        # 4. Validate output
```

### B. Validation Rules
```python
class DataValidator:
    """Validate spatial data before processing."""
    
    def validate_parcels(self, gdf: gpd.GeoDataFrame) -> Tuple[bool, Dict]:
        """
        Validate parcel data.
        
        Args:
            gdf: GeoDataFrame to validate
            
        Returns:
            Tuple of (is_valid, validation_results)
        """
        # Validation steps:
        # 1. Check required fields
        # 2. Validate geometry
        # 3. Check data types
        # 4. Verify spatial reference
```

## 3. Processing Framework

### A. Chunked Processing Configuration
```python
chunked_processor = ChunkedProcessor(
    chunk_size=10000,
    max_workers=4,
    memory_limit_mb=1000,
    srid=2264
)
```

### B. Error Recovery
```python
class TransactionManager:
    """Manage database transactions with recovery."""
    
    def process_with_recovery(
        self,
        data: gpd.GeoDataFrame,
        process_func: Callable,
        checkpoint_interval: int = 1000
    ) -> None:
        """
        Process data with transaction management and recovery.
        
        Args:
            data: Data to process
            process_func: Processing function
            checkpoint_interval: Records per checkpoint
        """
        # Implementation steps:
        # 1. Create checkpoints
        # 2. Process in transactions
        # 3. Handle failures
        # 4. Clean up checkpoints
```

## 4. Implementation Timeline

### Week 1: Database Setup
- [ ] Create database schemas
- [ ] Set up indices
- [ ] Create views
- [ ] Test database performance

### Week 2: Data Import
- [ ] Implement GDB processing
- [ ] Set up validation
- [ ] Create import procedures
- [ ] Test with sample data

### Week 3: Processing Framework
- [ ] Complete chunked processing
- [ ] Implement monitoring
- [ ] Set up error recovery
- [ ] Add progress tracking

### Week 4: Testing and Documentation
- [ ] Write integration tests
- [ ] Performance testing
- [ ] Update documentation
- [ ] User acceptance testing

## 5. Monitoring and Metrics

### A. Processing Metrics
- Records processed per second
- Memory usage per chunk
- Processing time distribution
- Error rates

### B. Database Metrics
- Query performance
- Index usage
- Transaction throughput
- Storage utilization

### C. Data Quality Metrics
- Validation success rate
- Geometry validity percentage
- Field completeness
- Relationship integrity

## 6. Success Criteria

### A. Performance
- [ ] Process 100,000 records in under 1 hour
- [ ] Memory usage below 2GB
- [ ] Query response time under 1 second
- [ ] Zero data loss during processing

### B. Data Quality
- [ ] 100% geometry validity
- [ ] All required fields present
- [ ] Correct spatial relationships
- [ ] Complete audit trail

### C. Reliability
- [ ] Automatic error recovery
- [ ] Transaction consistency
- [ ] No data corruption
- [ ] Complete logging
``` 