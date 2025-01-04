"""Utility script to take inventory of project files and organize them."""
from pathlib import Path
import shutil
import logging
from typing import List, Dict
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProjectInventory:
    """Manages project file inventory and organization."""
    
    STANDARD_DIRS = {
        'data/raw': 'Original, immutable data files',
        'data/processed': 'Cleaned and processed data',
        'data/interim': 'Intermediate data files',
        'src/models': 'Data models and analysis',
        'src/processing': 'Data processing scripts',
        'src/analysis': 'Analysis scripts',
        'src/visualization': 'Visualization code',
        'tests': 'Test files',
        'docs': 'Documentation',
        'notebooks': 'Jupyter notebooks',
        'config': 'Configuration files',
        'results': 'Output files and figures'
    }

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.inventory: Dict[str, List[Path]] = {}

    def create_standard_structure(self) -> None:
        """Create the standard project directory structure."""
        for dir_path in self.STANDARD_DIRS:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {full_path}")

    def take_inventory(self) -> Dict[str, List[Path]]:
        """Take inventory of all files in the project."""
        for item in self.project_root.rglob('*'):
            if item.is_file():
                ext = item.suffix.lower()
                if ext not in self.inventory:
                    self.inventory[ext] = []
                self.inventory[ext].append(item)
        return self.inventory

    def suggest_organization(self) -> Dict[str, List[str]]:
        """Suggest organization for files based on their extensions."""
        suggestions = {}
        
        # Define file type mappings
        file_mappings = {
            '.csv': 'data/raw',
            '.xlsx': 'data/raw',
            '.shp': 'data/raw',
            '.py': 'src',
            '.ipynb': 'notebooks',
            '.md': 'docs',
            '.json': 'config',
            '.yaml': 'config',
            '.txt': 'docs'
        }

        for ext, files in self.inventory.items():
            target_dir = file_mappings.get(ext, 'misc')
            if target_dir not in suggestions:
                suggestions[target_dir] = []
            suggestions[target_dir].extend([str(f) for f in files])

        return suggestions

    def generate_report(self) -> str:
        """Generate a report of the file inventory."""
        report = ["# Project File Inventory\n"]
        
        for ext, files in self.inventory.items():
            report.append(f"\n## {ext} files ({len(files)})")
            for f in files:
                report.append(f"- {f.relative_to(self.project_root)}")

        return "\n".join(report) 