"""
Monitor temporary files and their usage in the heirs property project.
"""
import os
import time
import logging
import psutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/data/heirs-property/logs/temp_monitor.log'),
        logging.StreamHandler()
    ]
)

class TempFileMonitor:
    """Monitor and manage temporary files."""
    
    def __init__(
        self,
        base_path: str = "/data/heirs-property",
        temp_dirs: Optional[List[str]] = None,
        size_threshold_mb: float = 100.0,
        age_threshold_hours: int = 24
    ):
        """Initialize the temporary file monitor.
        
        Args:
            base_path: Base path for heirs property data
            temp_dirs: List of directories to monitor (relative to base_path)
            size_threshold_mb: Alert threshold for individual files (MB)
            age_threshold_hours: Alert threshold for file age
        """
        self.base_path = Path(base_path)
        self.logger = logging.getLogger(__name__)
        self.temp_dirs = temp_dirs or [
            'interim/gis',
            'interim/ndvi',
            'interim/fia',
            'processed/parquet/temp',
            'processed/vectors/temp',
            'processed/rasters/temp'
        ]
        self.size_threshold = size_threshold_mb * 1024 * 1024  # Convert to bytes
        self.age_threshold = timedelta(hours=age_threshold_hours)
        
        # Ensure temp directories exist
        self._ensure_temp_dirs()
    
    def _ensure_temp_dirs(self) -> None:
        """Ensure all temporary directories exist."""
        for temp_dir in self.temp_dirs:
            dir_path = self.base_path / temp_dir
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Ensured temp directory exists: {dir_path}")
    
    def scan_temp_files(self) -> Dict[str, List[Dict]]:
        """Scan temporary directories for files.
        
        Returns:
            Dictionary mapping directory to list of file information
        """
        results = {}
        
        for temp_dir in self.temp_dirs:
            dir_path = self.base_path / temp_dir
            files = []
            
            if not dir_path.exists():
                continue
                
            for file_path in dir_path.glob('**/*'):
                if not file_path.is_file():
                    continue
                    
                stats = file_path.stat()
                age = datetime.now() - datetime.fromtimestamp(stats.st_mtime)
                size_mb = stats.st_size / (1024 * 1024)
                
                file_info = {
                    'path': str(file_path),
                    'size_mb': size_mb,
                    'age_hours': age.total_seconds() / 3600,
                    'last_modified': datetime.fromtimestamp(stats.st_mtime).isoformat()
                }
                
                # Check thresholds
                if stats.st_size > self.size_threshold:
                    self.logger.warning(
                        f"Large temp file detected: {file_path} "
                        f"({size_mb:.1f} MB)"
                    )
                
                if age > self.age_threshold:
                    self.logger.warning(
                        f"Old temp file detected: {file_path} "
                        f"({age.total_seconds() / 3600:.1f} hours old)"
                    )
                
                files.append(file_info)
            
            results[temp_dir] = files
        
        return results
    
    def cleanup_old_files(self, max_age_hours: int = 24) -> List[Path]:
        """Clean up old temporary files.
        
        Args:
            max_age_hours: Maximum age of files to keep
            
        Returns:
            List of cleaned up file paths
        """
        cleaned_files = []
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        
        for temp_dir in self.temp_dirs:
            dir_path = self.base_path / temp_dir
            if not dir_path.exists():
                continue
            
            for file_path in dir_path.glob('**/*'):
                if not file_path.is_file():
                    continue
                
                if datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff:
                    try:
                        file_path.unlink()
                        cleaned_files.append(file_path)
                        self.logger.info(f"Cleaned up old temp file: {file_path}")
                    except Exception as e:
                        self.logger.error(f"Error cleaning up {file_path}: {str(e)}")
        
        return cleaned_files
    
    def get_temp_usage_report(self) -> Dict:
        """Generate a comprehensive temporary file usage report.
        
        Returns:
            Dictionary containing usage statistics
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_size_mb': 0,
            'total_files': 0,
            'directories': {},
            'alerts': []
        }
        
        # Scan all temp directories
        scan_results = self.scan_temp_files()
        
        for temp_dir, files in scan_results.items():
            dir_stats = {
                'file_count': len(files),
                'total_size_mb': sum(f['size_mb'] for f in files),
                'oldest_file_hours': max([f['age_hours'] for f in files], default=0),
                'largest_file_mb': max([f['size_mb'] for f in files], default=0)
            }
            
            report['directories'][temp_dir] = dir_stats
            report['total_size_mb'] += dir_stats['total_size_mb']
            report['total_files'] += dir_stats['file_count']
        
        # Add disk usage information
        try:
            usage = psutil.disk_usage(str(self.base_path))
            report['disk_usage'] = {
                'total_gb': usage.total / (1024**3),
                'used_gb': usage.used / (1024**3),
                'free_gb': usage.free / (1024**3),
                'percent': usage.percent
            }
        except Exception as e:
            self.logger.error(f"Error getting disk usage: {str(e)}")
        
        return report

if __name__ == "__main__":
    # Example usage
    monitor = TempFileMonitor()
    
    # Generate report
    report = monitor.get_temp_usage_report()
    
    print("\nTemporary File Usage Report:")
    print(f"Total Size: {report['total_size_mb']:.1f} MB")
    print(f"Total Files: {report['total_files']}")
    
    print("\nBy Directory:")
    for dir_name, stats in report['directories'].items():
        print(f"\n{dir_name}:")
        print(f"  Files: {stats['file_count']}")
        print(f"  Size: {stats['total_size_mb']:.1f} MB")
        print(f"  Oldest File: {stats['oldest_file_hours']:.1f} hours")
        print(f"  Largest File: {stats['largest_file_mb']:.1f} MB")
    
    if 'disk_usage' in report:
        print("\nDisk Usage:")
        print(f"Total: {report['disk_usage']['total_gb']:.1f} GB")
        print(f"Used: {report['disk_usage']['used_gb']:.1f} GB ({report['disk_usage']['percent']}%)")
        print(f"Free: {report['disk_usage']['free_gb']:.1f} GB") 