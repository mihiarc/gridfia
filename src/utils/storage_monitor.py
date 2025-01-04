"""
Storage monitoring utility for the Heirs Property Analysis project.
"""
import os
import psutil
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/data/heirs-property/logs/storage_monitor.log'),
        logging.StreamHandler()
    ]
)

class StorageMonitor:
    """Monitor storage usage and manage cleanup operations."""
    
    def __init__(
        self,
        base_path: str = "/data/heirs-property",
        thresholds: Optional[Dict[str, float]] = None
    ):
        """Initialize the storage monitor.
        
        Args:
            base_path: Base path for heirs property data
            thresholds: Dictionary of path:threshold pairs
        """
        self.base_path = Path(base_path)
        self.logger = logging.getLogger(__name__)
        self.thresholds = thresholds or {
            '/data': 80.0,  # Alert at 80% usage
            '/': 90.0,     # Alert at 90% usage
            str(base_path): 70.0  # Alert at 70% usage
        }
        
        # Ensure log directory exists
        os.makedirs(self.base_path / "logs", exist_ok=True)
    
    def check_usage(self) -> Dict[str, float]:
        """Check storage usage against thresholds.
        
        Returns:
            Dictionary of path:usage pairs
        """
        usage_stats = {}
        for path, threshold in self.thresholds.items():
            try:
                usage = psutil.disk_usage(path)
                usage_stats[path] = usage.percent
                if usage.percent >= threshold:
                    self.alert(path, usage.percent)
            except Exception as e:
                self.logger.error(f"Error checking usage for {path}: {str(e)}")
        return usage_stats
    
    def alert(self, path: str, usage: float) -> None:
        """Send alert for high storage usage.
        
        Args:
            path: Path that triggered the alert
            usage: Current usage percentage
        """
        message = f"Storage alert: {path} is at {usage:.1f}% usage"
        self.logger.warning(message)
        
        # Log to separate alerts file
        with open(self.base_path / "logs/alerts.log", "a") as f:
            f.write(f"{datetime.now().isoformat()}: {message}\n")
    
    def cleanup_interim_data(self, days: int = 7) -> List[Path]:
        """Clean interim data older than specified days.
        
        Args:
            days: Number of days to keep data
            
        Returns:
            List of cleaned up files
        """
        cleaned_files = []
        interim_path = self.base_path / "interim"
        cutoff = datetime.now() - timedelta(days=days)
        
        try:
            for file in interim_path.glob("**/*"):
                if file.is_file() and file.stat().st_mtime < cutoff.timestamp():
                    file.unlink()
                    cleaned_files.append(file)
                    self.logger.info(f"Cleaned up old interim file: {file}")
        except Exception as e:
            self.logger.error(f"Error during interim cleanup: {str(e)}")
        
        return cleaned_files
    
    def cleanup_processed_versions(self, keep_versions: int = 3) -> Dict[str, List[Path]]:
        """Keep only specified number of latest versions of processed data.
        
        Args:
            keep_versions: Number of versions to keep
            
        Returns:
            Dictionary of category:cleaned_files pairs
        """
        cleaned_files = {}
        processed_path = self.base_path / "processed"
        
        for category in ['parquet', 'vectors', 'rasters']:
            category_path = processed_path / category
            if not category_path.exists():
                continue
                
            try:
                files = sorted(
                    category_path.glob("*"),
                    key=lambda x: x.stat().st_mtime,
                    reverse=True
                )
                
                # Keep only specified number of versions
                for old_file in files[keep_versions:]:
                    old_file.unlink()
                    cleaned_files.setdefault(category, []).append(old_file)
                    self.logger.info(f"Cleaned up old {category} file: {old_file}")
            except Exception as e:
                self.logger.error(f"Error cleaning up {category}: {str(e)}")
        
        return cleaned_files
    
    def get_storage_report(self) -> Dict:
        """Generate a comprehensive storage report.
        
        Returns:
            Dictionary containing storage statistics
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'usage': self.check_usage(),
            'directory_sizes': {},
            'alerts': []
        }
        
        # Get directory sizes
        for dir_name in ['raw', 'processed', 'interim', 'postgres']:
            dir_path = self.base_path / dir_name
            if dir_path.exists():
                total_size = sum(
                    f.stat().st_size for f in dir_path.glob('**/*') if f.is_file()
                )
                report['directory_sizes'][dir_name] = total_size / (1024 * 1024 * 1024)  # GB
        
        # Check for recent alerts
        if (self.base_path / "logs/alerts.log").exists():
            with open(self.base_path / "logs/alerts.log") as f:
                report['alerts'] = [
                    line.strip() for line in f.readlines()[-10:]  # Last 10 alerts
                ]
        
        return report

if __name__ == "__main__":
    # Example usage
    monitor = StorageMonitor()
    
    # Check usage
    usage_stats = monitor.check_usage()
    print("Storage Usage:")
    for path, usage in usage_stats.items():
        print(f"{path}: {usage:.1f}%")
    
    # Clean up old data
    cleaned_interim = monitor.cleanup_interim_data()
    cleaned_processed = monitor.cleanup_processed_versions()
    
    # Generate report
    report = monitor.get_storage_report()
    print("\nStorage Report:")
    print(f"Directory Sizes (GB):")
    for dir_name, size in report['directory_sizes'].items():
        print(f"{dir_name}: {size:.2f}GB") 