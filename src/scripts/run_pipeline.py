#!/usr/bin/env python3
"""
Data pipeline runner for the Heirs Property Analysis project.
Orchestrates the processing of parcel, heirs property, and related data.
"""
import os
from pathlib import Path
import logging
from typing import Optional, Dict
import sys
from datetime import datetime
import json

from dotenv import load_dotenv

from processing.parcel_processor import ParcelProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Required fields for each dataset type
PARCEL_REQUIRED_FIELDS = [
    'geometry',
    'parcel_id',
    'area',
    'county'
]

HEIRS_PROPERTY_REQUIRED_FIELDS = [
    'geometry',
    'hp_id',
    'area',
    'county',
    'status'
]

# Processing configuration
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '10000'))
MAX_WORKERS = int(os.getenv('MAX_WORKERS', '4'))
MEMORY_LIMIT_MB = int(os.getenv('MEMORY_LIMIT_MB', '1000'))

# Monitoring configuration
RESOURCE_INTERVAL = int(os.getenv('RESOURCE_INTERVAL', '30'))
CPU_THRESHOLD = float(os.getenv('CPU_THRESHOLD', '80.0'))
MEMORY_THRESHOLD = float(os.getenv('MEMORY_THRESHOLD', '80.0'))
DISK_THRESHOLD = float(os.getenv('DISK_THRESHOLD', '80.0'))

def get_db_connection() -> str:
    """Get database connection string from environment variables."""
    load_dotenv()
    
    # Required environment variables
    required_vars = [
        'POSTGRES_USER',
        'POSTGRES_PASSWORD',
        'POSTGRES_HOST',
        'POSTGRES_PORT',
        'POSTGRES_DB'
    ]
    
    # Check for missing environment variables
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Construct connection string
    return (
        f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
        f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
    )

def setup_validation_dir(data_dir: Path) -> Path:
    """Set up validation report directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    validation_dir = data_dir / "validation" / timestamp
    validation_dir.mkdir(parents=True, exist_ok=True)
    return validation_dir

def setup_stats_dir(data_dir: Path) -> Path:
    """Set up processing stats directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_dir = data_dir / "stats" / timestamp
    stats_dir.mkdir(parents=True, exist_ok=True)
    return stats_dir

def setup_checkpoint_dir(data_dir: Path) -> Path:
    """Set up checkpoint directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = data_dir / "checkpoints" / timestamp
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir

def setup_metrics_dir(data_dir: Path) -> Path:
    """Set up metrics directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_dir = data_dir / "metrics" / timestamp
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return metrics_dir

def process_parcel_data(
    processor: ParcelProcessor,
    data_dir: Path,
    validation_dir: Path,
    stats_dir: Path
) -> bool:
    """Process parcel data from both Parquet and GDB sources."""
    success = True
    
    # Process main parcel dataset
    parcel_path = data_dir / "raw" / "nc-parcels.parquet"
    if parcel_path.exists():
        logger.info("Processing main parcel dataset...")
        validation_path = validation_dir / "parcels_validation.json"
        stats_path = stats_dir / "parcels_stats.json"
        if not processor.process_parquet(
            parcel_path,
            "parcels",
            validation_report_path=validation_path,
            stats_path=stats_path
        ):
            success = False
            logger.error("Failed to process main parcel dataset")
    else:
        logger.warning(f"Parcel file not found: {parcel_path}")
        success = False
    
    # Process parcels within 1 mile
    nearby_path = data_dir / "raw" / "parcels_within_1_mile.parquet"
    if nearby_path.exists():
        logger.info("Processing nearby parcels dataset...")
        validation_path = validation_dir / "nearby_parcels_validation.json"
        stats_path = stats_dir / "nearby_parcels_stats.json"
        if not processor.process_parquet(
            nearby_path,
            "nearby_parcels",
            validation_report_path=validation_path,
            stats_path=stats_path
        ):
            success = False
            logger.error("Failed to process nearby parcels dataset")
    else:
        logger.warning(f"Nearby parcels file not found: {nearby_path}")
        success = False
    
    # Process GDB data if available
    gdb_path = data_dir / "raw" / "NC.gdb"
    if gdb_path.exists():
        logger.info("Processing GDB dataset...")
        validation_path = validation_dir / "nc_parcels_gdb_validation.json"
        stats_path = stats_dir / "nc_parcels_gdb_stats.json"
        if not processor.process_gdb(
            gdb_path,
            "parcels",
            "nc_parcels_gdb",
            validation_report_path=validation_path,
            stats_path=stats_path
        ):
            success = False
            logger.error("Failed to process GDB dataset")
    else:
        logger.warning(f"GDB file not found: {gdb_path}")
    
    return success

def process_heirs_property(
    processor: ParcelProcessor,
    data_dir: Path,
    validation_dir: Path,
    stats_dir: Path
) -> bool:
    """Process heirs property data."""
    success = True
    
    # Process heirs property dataset
    hp_path = data_dir / "raw" / "nc-hp_v2.parquet"
    if hp_path.exists():
        logger.info("Processing heirs property dataset...")
        validation_path = validation_dir / "heirs_properties_validation.json"
        stats_path = stats_dir / "heirs_properties_stats.json"
        if not processor.process_parquet(
            hp_path,
            "heirs_properties",
            validation_report_path=validation_path,
            stats_path=stats_path
        ):
            success = False
            logger.error("Failed to process heirs property dataset")
    else:
        logger.warning(f"Heirs property file not found: {hp_path}")
        success = False
    
    # Process GDB data if available
    hp_gdb_path = data_dir / "raw" / "HP_Deliverables.gdb"
    if hp_gdb_path.exists():
        logger.info("Processing HP GDB dataset...")
        validation_path = validation_dir / "hp_gdb_validation.json"
        stats_path = stats_dir / "hp_gdb_stats.json"
        if not processor.process_gdb(
            hp_gdb_path,
            "heirs_properties",
            "hp_gdb",
            validation_report_path=validation_path,
            stats_path=stats_path
        ):
            success = False
            logger.error("Failed to process HP GDB dataset")
    else:
        logger.warning(f"HP GDB file not found: {hp_gdb_path}")
    
    return success

def generate_pipeline_report(
    data_dir: Path,
    validation_dir: Path,
    stats_dir: Path,
    metrics_dir: Path,
    output_path: Optional[Path] = None
) -> Dict:
    """Generate a comprehensive pipeline report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "pipeline_status": {
            "validation_dir": str(validation_dir),
            "stats_dir": str(stats_dir),
            "metrics_dir": str(metrics_dir)
        },
        "validation_reports": {},
        "processing_stats": {},
        "monitoring_metrics": {}
    }
    
    # Collect validation reports
    for report_file in validation_dir.glob("*.json"):
        with open(report_file) as f:
            report["validation_reports"][report_file.stem] = json.load(f)
    
    # Collect processing stats
    for stats_file in stats_dir.glob("*.json"):
        with open(stats_file) as f:
            report["processing_stats"][stats_file.stem] = json.load(f)
    
    # Collect monitoring metrics
    for metrics_file in metrics_dir.glob("*.json"):
        with open(metrics_file) as f:
            report["monitoring_metrics"][metrics_file.stem] = json.load(f)
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    return report

def main():
    """Main pipeline execution function."""
    try:
        # Get database connection
        db_connection = get_db_connection()
        
        # Set up data directory
        data_dir = Path(__file__).parents[2] / "data"
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Set up output directories
        validation_dir = setup_validation_dir(data_dir)
        stats_dir = setup_stats_dir(data_dir)
        checkpoint_dir = setup_checkpoint_dir(data_dir)
        metrics_dir = setup_metrics_dir(data_dir)
        
        logger.info(f"Validation reports will be saved to: {validation_dir}")
        logger.info(f"Processing stats will be saved to: {stats_dir}")
        logger.info(f"Transaction checkpoints will be saved to: {checkpoint_dir}")
        logger.info(f"Monitoring metrics will be saved to: {metrics_dir}")
        
        # Initialize processors with required fields
        parcel_processor = ParcelProcessor(
            db_connection=db_connection,
            schema="public",
            srid=2264,
            required_fields=PARCEL_REQUIRED_FIELDS,
            chunk_size=CHUNK_SIZE,
            max_workers=MAX_WORKERS,
            memory_limit_mb=MEMORY_LIMIT_MB,
            checkpoint_dir=checkpoint_dir / "parcels",
            metrics_dir=metrics_dir / "parcels"
        )
        
        hp_processor = ParcelProcessor(
            db_connection=db_connection,
            schema="public",
            srid=2264,
            required_fields=HEIRS_PROPERTY_REQUIRED_FIELDS,
            chunk_size=CHUNK_SIZE,
            max_workers=MAX_WORKERS,
            memory_limit_mb=MEMORY_LIMIT_MB,
            checkpoint_dir=checkpoint_dir / "heirs_property",
            metrics_dir=metrics_dir / "heirs_property"
        )
        
        # Process datasets
        parcel_success = process_parcel_data(
            parcel_processor,
            data_dir,
            validation_dir,
            stats_dir
        )
        hp_success = process_heirs_property(
            hp_processor,
            data_dir,
            validation_dir,
            stats_dir
        )
        
        # Generate pipeline report
        report_path = data_dir / "reports" / f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = generate_pipeline_report(
            data_dir,
            validation_dir,
            stats_dir,
            metrics_dir,
            report_path
        )
        
        # Report results
        if parcel_success and hp_success:
            logger.info("Pipeline completed successfully")
            logger.info(f"Validation reports available at: {validation_dir}")
            logger.info(f"Processing stats available at: {stats_dir}")
            logger.info(f"Transaction checkpoints available at: {checkpoint_dir}")
            logger.info(f"Monitoring metrics available at: {metrics_dir}")
            logger.info(f"Pipeline report available at: {report_path}")
            sys.exit(0)
        else:
            logger.error("Pipeline completed with errors")
            logger.error(f"Check validation reports at: {validation_dir}")
            logger.error(f"Check processing stats at: {stats_dir}")
            logger.error(f"Check transaction checkpoints at: {checkpoint_dir}")
            logger.error(f"Check monitoring metrics at: {metrics_dir}")
            logger.error(f"Check pipeline report at: {report_path}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 