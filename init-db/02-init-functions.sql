-- Function to calculate forest fragmentation metrics
CREATE OR REPLACE FUNCTION analysis.calculate_forest_fragmentation(
    target_parcel_id VARCHAR,
    ndvi_threshold DOUBLE PRECISION DEFAULT 0.5
)
RETURNS TABLE (
    forest_area_sqm DOUBLE PRECISION,
    forest_patches INTEGER,
    edge_density DOUBLE PRECISION,
    core_area_sqm DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    WITH forest_patches AS (
        SELECT 
            (ST_Dump(
                ST_CollectionExtract(
                    ST_Intersection(
                        p.geometry,
                        ST_Union(
                            CASE 
                                WHEN n.mean_ndvi >= ndvi_threshold THEN 
                                    ST_Buffer(p.geometry, 0)
                                ELSE NULL 
                            END
                        )
                    )
                )
            )).geom as patch_geom
        FROM 
            vector.parcels p
            JOIN analysis.ndvi_stats n ON p.parcel_id = n.parcel_id
        WHERE 
            p.parcel_id = target_parcel_id
    )
    SELECT 
        SUM(ST_Area(patch_geom::geography)) as forest_area_sqm,
        COUNT(*) as forest_patches,
        SUM(ST_Perimeter(patch_geom::geography)) / NULLIF(SUM(ST_Area(patch_geom::geography)), 0) as edge_density,
        SUM(ST_Area(ST_Buffer(patch_geom::geography, -30)::geometry)) as core_area_sqm
    FROM 
        forest_patches;
END;
$$ LANGUAGE plpgsql;

-- Function to analyze management patterns
CREATE OR REPLACE FUNCTION analysis.compare_management_patterns(
    target_parcel_id VARCHAR,
    buffer_meters DOUBLE PRECISION DEFAULT 1609.34,  -- 1 mile
    time_window_years INTEGER DEFAULT 5
)
RETURNS TABLE (
    comparison_type VARCHAR,
    mean_ndvi_diff DOUBLE PRECISION,
    forest_area_diff_pct DOUBLE PRECISION,
    sample_size INTEGER
) AS $$
BEGIN
    RETURN QUERY
    WITH neighbor_stats AS (
        SELECT 
            CASE 
                WHEN hp.hp_id IS NOT NULL THEN 'heir'
                ELSE 'non_heir'
            END as property_type,
            n.mean_ndvi,
            f.forest_area_sqm,
            p.area_sqm
        FROM 
            vector.parcels p
            LEFT JOIN vector.heirs_properties hp ON p.parcel_id = hp.parcel_id
            LEFT JOIN analysis.ndvi_stats n ON p.parcel_id = n.parcel_id
            CROSS JOIN LATERAL (
                SELECT forest_area_sqm 
                FROM analysis.calculate_forest_fragmentation(p.parcel_id)
            ) f
        WHERE 
            p.parcel_id IN (
                SELECT parcel_id 
                FROM vector.get_neighboring_parcels(target_parcel_id, buffer_meters)
            )
            AND n.capture_date >= (CURRENT_DATE - (time_window_years || ' years')::INTERVAL)
    )
    SELECT 
        property_type as comparison_type,
        AVG(mean_ndvi - (
            SELECT AVG(mean_ndvi) 
            FROM neighbor_stats
        )) as mean_ndvi_diff,
        AVG((forest_area_sqm / area_sqm) - (
            SELECT AVG(forest_area_sqm / area_sqm) 
            FROM neighbor_stats
        )) * 100 as forest_area_diff_pct,
        COUNT(*) as sample_size
    FROM 
        neighbor_stats
    GROUP BY 
        property_type;
END;
$$ LANGUAGE plpgsql;

-- Function to detect temporal changes
CREATE OR REPLACE FUNCTION analysis.detect_temporal_changes(
    target_parcel_id VARCHAR,
    start_date DATE,
    end_date DATE
)
RETURNS TABLE (
    period_start DATE,
    period_end DATE,
    ndvi_change DOUBLE PRECISION,
    forest_area_change_sqm DOUBLE PRECISION,
    change_significance DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    WITH time_series AS (
        SELECT 
            n.capture_date,
            n.mean_ndvi,
            f.forest_area_sqm,
            n.std_ndvi
        FROM 
            analysis.ndvi_stats n
            CROSS JOIN LATERAL (
                SELECT forest_area_sqm 
                FROM analysis.calculate_forest_fragmentation(target_parcel_id)
            ) f
        WHERE 
            n.parcel_id = target_parcel_id
            AND n.capture_date BETWEEN start_date AND end_date
        ORDER BY 
            n.capture_date
    ),
    changes AS (
        SELECT 
            lag(capture_date) OVER w as period_start,
            capture_date as period_end,
            mean_ndvi - lag(mean_ndvi) OVER w as ndvi_diff,
            forest_area_sqm - lag(forest_area_sqm) OVER w as area_diff,
            (mean_ndvi - lag(mean_ndvi) OVER w) / NULLIF(std_ndvi, 0) as significance
        FROM time_series
        WINDOW w AS (ORDER BY capture_date)
    )
    SELECT 
        period_start,
        period_end,
        ndvi_diff as ndvi_change,
        area_diff as forest_area_change_sqm,
        significance as change_significance
    FROM changes
    WHERE period_start IS NOT NULL;
END;
$$ LANGUAGE plpgsql;

-- Function to generate summary statistics
CREATE OR REPLACE FUNCTION analysis.generate_property_summary(
    target_parcel_id VARCHAR
)
RETURNS TABLE (
    property_type VARCHAR,
    total_area_sqm DOUBLE PRECISION,
    forest_cover_pct DOUBLE PRECISION,
    mean_ndvi DOUBLE PRECISION,
    edge_density DOUBLE PRECISION,
    neighbor_similarity_score DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    WITH property_data AS (
        SELECT 
            CASE 
                WHEN hp.hp_id IS NOT NULL THEN 'Heir Property'
                ELSE 'Standard Property'
            END as prop_type,
            p.area_sqm,
            n.mean_ndvi,
            f.forest_area_sqm,
            f.edge_density
        FROM 
            vector.parcels p
            LEFT JOIN vector.heirs_properties hp ON p.parcel_id = hp.parcel_id
            LEFT JOIN analysis.ndvi_stats n ON p.parcel_id = n.parcel_id
            CROSS JOIN LATERAL (
                SELECT * FROM analysis.calculate_forest_fragmentation(p.parcel_id)
            ) f
        WHERE 
            p.parcel_id = target_parcel_id
    ),
    neighbor_comparison AS (
        SELECT 
            AVG(
                CASE 
                    WHEN c.mean_ndvi_diff < 0 THEN 1 - ABS(c.mean_ndvi_diff)
                    ELSE 1 + c.mean_ndvi_diff
                END
            ) as similarity_score
        FROM 
            analysis.compare_management_patterns(target_parcel_id) c
    )
    SELECT 
        d.prop_type,
        d.area_sqm,
        (d.forest_area_sqm / d.area_sqm * 100) as forest_cover_pct,
        d.mean_ndvi,
        d.edge_density,
        nc.similarity_score as neighbor_similarity_score
    FROM 
        property_data d
        CROSS JOIN neighbor_comparison nc;
END;
$$ LANGUAGE plpgsql; 