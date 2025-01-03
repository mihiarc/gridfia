"""Script to run project inventory and generate report."""
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.file_inventory import ProjectInventory
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run the project inventory process."""
    # Get project root (assuming this script is in src/scripts)
    project_root = Path(__file__).parent.parent.parent
    
    # Create inventory manager
    inventory = ProjectInventory(project_root)
    
    # Create standard directory structure
    inventory.create_standard_structure()
    
    # Take inventory
    logger.info("Taking inventory of project files...")
    inventory.take_inventory()
    
    # Generate and save report
    report = inventory.generate_report()
    report_path = project_root / "docs" / "file_inventory.md"
    report_path.write_text(report)
    logger.info(f"Inventory report saved to {report_path}")
    
    # Get organization suggestions
    suggestions = inventory.suggest_organization()
    logger.info("\nSuggested file organization:")
    for target_dir, files in suggestions.items():
        logger.info(f"\n{target_dir}:")
        for f in files:
            logger.info(f"  - {f}")

if __name__ == "__main__":
    main() 