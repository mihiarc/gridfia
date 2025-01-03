"""
Pipeline monitoring module for resource tracking.
"""
import json
import time
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

import psutil

@dataclass
class ResourceMetrics:
    """System resource metrics."""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_percent: float
    disk_used_gb: float

@dataclass
class ProcessingMetrics:
    """Dataset processing metrics."""
    dataset_name: str
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    total_records: int = 0
    processed_records: int = 0
    failed_records: int = 0
    validation_errors: int = 0
    processing_errors: int = 0
    avg_processing_time: float = 0.0
    peak_memory_mb: float = 0.0
    memory_usage: Dict[str, float] = field(default_factory=dict)

class PipelineMonitor:
    """Monitors pipeline resources and performance."""
    
    def __init__(
        self,
        metrics_dir: Path,
        resource_interval: float = 5.0,
        alert_cpu_threshold: float = 80.0,
        alert_memory_threshold: float = 80.0,
        alert_disk_threshold: float = 80.0
    ):
        """Initialize monitor with configuration.
        
        Args:
            metrics_dir: Directory for storing metrics
            resource_interval: Interval for resource checks in seconds
            alert_cpu_threshold: CPU usage threshold for alerts
            alert_memory_threshold: Memory usage threshold for alerts
            alert_disk_threshold: Disk usage threshold for alerts
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        self.resource_interval = resource_interval
        self.alert_cpu_threshold = alert_cpu_threshold
        self.alert_memory_threshold = alert_memory_threshold
        self.alert_disk_threshold = alert_disk_threshold
        
        self.resource_metrics = []
        self.processing_metrics = {}
        
        self._monitoring = False
        self._monitor_thread = None
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_resources,
            daemon=True
        )
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
            self._monitor_thread = None
    
    def _monitor_resources(self):
        """Monitor system resources."""
        while self._monitoring:
            try:
                # Get CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Get memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_used_mb = memory.used / 1024 / 1024
                
                # Get disk usage
                disk = psutil.disk_usage('/')
                disk_percent = disk.percent
                disk_used_gb = disk.used / 1024 / 1024 / 1024
                
                # Create metrics
                metrics = ResourceMetrics(
                    timestamp=datetime.now().isoformat(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory_percent,
                    memory_used_mb=memory_used_mb,
                    disk_percent=disk_percent,
                    disk_used_gb=disk_used_gb
                )
                
                # Check thresholds
                if cpu_percent > self.alert_cpu_threshold:
                    self.logger.warning(
                        f"High CPU usage: {cpu_percent:.1f}%"
                    )
                if memory_percent > self.alert_memory_threshold:
                    self.logger.warning(
                        f"High memory usage: {memory_percent:.1f}%"
                    )
                if disk_percent > self.alert_disk_threshold:
                    self.logger.warning(
                        f"High disk usage: {disk_percent:.1f}%"
                    )
                
                # Store metrics
                self.resource_metrics.append(metrics)
                
                # Wait for next check
                time.sleep(self.resource_interval)
                
            except Exception as e:
                self.logger.error(f"Error monitoring resources: {str(e)}")
                time.sleep(self.resource_interval)
    
    def start_processing_metrics(
        self,
        dataset_name: str,
        total_records: int
    ) -> ProcessingMetrics:
        """Initialize processing metrics for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            total_records: Total number of records
            
        Returns:
            New ProcessingMetrics instance
        """
        metrics = ProcessingMetrics(
            dataset_name=dataset_name,
            total_records=total_records
        )
        self.processing_metrics[dataset_name] = metrics
        return metrics
    
    def update_processing_metrics(
        self,
        dataset_name: str,
        processed_records: Optional[int] = None,
        failed_records: Optional[int] = None,
        validation_errors: Optional[int] = None,
        processing_errors: Optional[int] = None,
        processing_time: Optional[float] = None,
        memory_mb: Optional[float] = None
    ):
        """Update processing metrics for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            processed_records: Number of processed records
            failed_records: Number of failed records
            validation_errors: Number of validation errors
            processing_errors: Number of processing errors
            processing_time: Average processing time
            memory_mb: Memory usage in MB
        """
        metrics = self.processing_metrics.get(dataset_name)
        if not metrics:
            return
        
        if processed_records is not None:
            metrics.processed_records = processed_records
        if failed_records is not None:
            metrics.failed_records = failed_records
        if validation_errors is not None:
            metrics.validation_errors = validation_errors
        if processing_errors is not None:
            metrics.processing_errors = processing_errors
        if processing_time is not None:
            metrics.avg_processing_time = processing_time
        if memory_mb is not None:
            metrics.peak_memory_mb = max(
                metrics.peak_memory_mb,
                memory_mb
            )
            metrics.memory_usage[datetime.now().isoformat()] = memory_mb
    
    def get_processing_metrics(
        self,
        dataset_name: str
    ) -> Optional[ProcessingMetrics]:
        """Get processing metrics for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            ProcessingMetrics instance or None if not found
        """
        return self.processing_metrics.get(dataset_name)
    
    def get_latest_resource_metrics(self) -> Optional[ResourceMetrics]:
        """Get latest resource metrics.
        
        Returns:
            Latest ResourceMetrics instance or None
        """
        return self.resource_metrics[-1] if self.resource_metrics else None
    
    def save_metrics(self, output_path: Optional[Path] = None):
        """Save metrics to file.
        
        Args:
            output_path: Optional path for metrics file
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "resource_metrics": [
                {
                    "timestamp": m.timestamp,
                    "cpu_percent": m.cpu_percent,
                    "memory_percent": m.memory_percent,
                    "memory_used_mb": m.memory_used_mb,
                    "disk_percent": m.disk_percent,
                    "disk_used_gb": m.disk_used_gb
                }
                for m in self.resource_metrics
            ],
            "processing_metrics": {
                name: {
                    "dataset_name": m.dataset_name,
                    "start_time": m.start_time,
                    "total_records": m.total_records,
                    "processed_records": m.processed_records,
                    "failed_records": m.failed_records,
                    "validation_errors": m.validation_errors,
                    "processing_errors": m.processing_errors,
                    "avg_processing_time": m.avg_processing_time,
                    "peak_memory_mb": m.peak_memory_mb,
                    "memory_usage": m.memory_usage
                }
                for name, m in self.processing_metrics.items()
            }
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=2)
        else:
            metrics_path = (
                self.metrics_dir / 
                f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
    
    def generate_report(
        self,
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Generate monitoring report.
        
        Args:
            output_path: Optional path for report file
            
        Returns:
            Report dictionary
        """
        # Calculate resource metrics
        resource_metrics = {
            "averages": {
                "cpu_percent": sum(
                    m.cpu_percent for m in self.resource_metrics
                ) / len(self.resource_metrics) if self.resource_metrics else 0,
                "memory_percent": sum(
                    m.memory_percent for m in self.resource_metrics
                ) / len(self.resource_metrics) if self.resource_metrics else 0,
                "disk_percent": sum(
                    m.disk_percent for m in self.resource_metrics
                ) / len(self.resource_metrics) if self.resource_metrics else 0
            },
            "peaks": {
                "cpu_percent": max(
                    (m.cpu_percent for m in self.resource_metrics),
                    default=0
                ),
                "memory_percent": max(
                    (m.memory_percent for m in self.resource_metrics),
                    default=0
                ),
                "disk_percent": max(
                    (m.disk_percent for m in self.resource_metrics),
                    default=0
                )
            }
        }
        
        # Calculate processing metrics
        processing_metrics = {
            name: {
                "total_records": m.total_records,
                "processed_records": m.processed_records,
                "failed_records": m.failed_records,
                "validation_errors": m.validation_errors,
                "processing_errors": m.processing_errors,
                "avg_processing_time": m.avg_processing_time,
                "peak_memory_mb": m.peak_memory_mb
            }
            for name, m in self.processing_metrics.items()
        }
        
        # Create report
        report = {
            "timestamp": datetime.now().isoformat(),
            "resource_metrics": resource_metrics,
            "processing_metrics": processing_metrics,
            "summary": {
                "total_duration": (
                    time.time() - float(self.resource_metrics[0].timestamp)
                    if self.resource_metrics else 0
                ),
                "total_records": sum(
                    m.total_records 
                    for m in self.processing_metrics.values()
                ),
                "total_processed": sum(
                    m.processed_records 
                    for m in self.processing_metrics.values()
                ),
                "total_failed": sum(
                    m.failed_records 
                    for m in self.processing_metrics.values()
                )
            }
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate processing summary.
        
        Returns:
            Summary dictionary
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "datasets": list(self.processing_metrics.keys()),
            "total_records": sum(
                m.total_records for m in self.processing_metrics.values()
            ),
            "total_processed": sum(
                m.processed_records for m in self.processing_metrics.values()
            ),
            "total_failed": sum(
                m.failed_records for m in self.processing_metrics.values()
            ),
            "total_errors": sum(
                m.validation_errors + m.processing_errors
                for m in self.processing_metrics.values()
            )
        } 