"""Script to reorganize project files into standard structure."""
import shutil
import logging
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProjectReorganizer:
    """Handles reorganization of project files into standard structure."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.data_raw = project_root / "src" / "data" / "raw"
        self.data_processed = project_root / "src" / "data" / "processed"
        self.notebooks_dir = project_root / "notebooks"

    def create_directories(self):
        """Create necessary directories if they don't exist."""
        dirs = [
            self.data_raw / "gis",
            self.data_raw / "ndvi",
            self.data_processed,
            self.notebooks_dir,
            self.project_root / "docs",
            self.project_root / "src" / "visualization",
            self.project_root / "src" / "processing"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")

    def move_files(self):
        """Move files to their appropriate locations."""
        # Move GIS data
        gis_files = [
            ('NC.gdb', 'src/data/raw/gis'),
            ('HP_Deliverables.gdb', 'src/data/raw/gis'),
            ('HP_Deliverables.gdb.zip', 'src/data/raw/gis')
        ]

        # Move parquet files
        parquet_files = [
            ('nc-parcels.parquet', 'src/data/processed'),
            ('nc-hp_v2.parquet', 'src/data/processed'),
            ('parcels_within_1_mile.parquet', 'src/data/processed'),
            ('nc-hp.parquet', 'src/data/processed')
        ]

        # Move notebooks
        notebook_files = [
            ('convert-gdb-to-parquet.ipynb', 'notebooks'),
            ('hp-fia-plots.ipynb', 'notebooks'),
            ('convert-hp-to-parquet.ipynb', 'notebooks'),
            ('hp-fia-plots-part2.ipynb', 'notebooks')
        ]

        # Move NDVI files
        ndvi_dirs = [
            ('ndvi_Heirs-Vance_County', 'src/data/raw/ndvi'),
            ('ndvi_layers', 'src/data/raw/ndvi')
        ]

        # Move Python files
        python_files = [
            ('ndvi_visualization.py', 'src/visualization'),
            ('create_hp_neighbors.py', 'src/processing')
        ]

        # Move documentation
        doc_files = [
            ('HP_Properties_NC_ReadME.docx', 'docs')
        ]

        # Execute moves
        all_moves = (
            gis_files + parquet_files + notebook_files + 
            python_files + doc_files
        )

        for src_name, dest_dir in all_moves:
            src_path = self.project_root / src_name
            dest_path = self.project_root / dest_dir / src_name
            
            if src_path.exists():
                try:
                    if src_path.is_dir():
                        if dest_path.exists():
                            shutil.rmtree(dest_path)
                        shutil.copytree(src_path, dest_path)
                        shutil.rmtree(src_path)
                    else:
                        shutil.move(str(src_path), str(dest_path))
                    logger.info(f"Moved {src_name} to {dest_dir}")
                except Exception as e:
                    logger.error(f"Error moving {src_name}: {e}")
            else:
                logger.warning(f"Source file not found: {src_name}")

        # Handle NDVI directories
        for src_dir, dest_base in ndvi_dirs:
            src_path = self.project_root / src_dir
            if src_path.exists():
                try:
                    # Move all .tif files
                    for tif_file in src_path.rglob('*.tif'):
                        dest_path = self.project_root / dest_base / tif_file.name
                        shutil.move(str(tif_file), str(dest_path))
                        logger.info(f"Moved {tif_file.name} to {dest_base}")
                except Exception as e:
                    logger.error(f"Error moving NDVI files from {src_dir}: {e}")

def main():
    """Run the reorganization process."""
    project_root = Path(__file__).parent.parent.parent
    
    logger.info("Starting project reorganization...")
    reorganizer = ProjectReorganizer(project_root)
    
    logger.info("Creating directory structure...")
    reorganizer.create_directories()
    
    logger.info("Moving files to new locations...")
    reorganizer.move_files()
    
    logger.info("Project reorganization complete!")

if __name__ == "__main__":
    main() 