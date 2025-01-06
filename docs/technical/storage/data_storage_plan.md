# Data Storage Management Plan

## Overview

This document outlines the strategy for managing data storage in the Heirs Property Analysis project, focusing on efficient use of available storage resources during development.

## 1. Storage Architecture

### Current Storage Analysis
```mermaid
graph LR
    subgraph Volumes
        R[Root /]
        D[/data]
        H[/home]
    end
    
    subgraph Usage
        R --> |94G Total|RU[88G Used]
        D --> |3.6T Total|DU[3.2T Used]
        H --> |916G Total|HU[199G Used]
    end
```

### Project Storage Requirements
| Component | Current Size | Projected Size | Growth Rate |
|-----------|--------------|----------------|-------------|
| Raw Data | 6.4G | ~10G | Low |
| Processed Data | 15G | ~20G | Medium |
| Database | ~5G | ~15G | Medium |
| Interim Files | ~2G | ~5G | High |

## 2. Directory Structure

### Base Directory Layout
```
/data/heirs-property/
├── raw/
│   ├── gis/
│   │   ├── NC.gdb
│   │   └── HP_Deliverables.gdb
│   ├── ndvi/
│   │   └── naip/
│   └── fia/
│       ├── nc-fia-plots.csv
│       └── nc-plots.csv
├── processed/
│   ├── parquet/
│   ├── vectors/
│   └── rasters/
├── interim/
│   ├── temp/
│   └── cache/
└── postgres/
    └── data/
```

### Symbolic Link Structure
```mermaid
graph TD
    A[Project Root] -->|symlink| B[data/]
    B --> C[/data/heirs-property]
    C --> D[raw]
    C --> E[processed]
    C --> F[interim]
    C --> G[postgres]
```

## 3. Implementation Plan

### Phase 1: Directory Setup
```bash
# Base directory structure
mkdir -p /data/heirs-property/{raw,processed,interim}/{gis,ndvi,fia}
mkdir -p /data/heirs-property/processed/{parquet,vectors,rasters}
mkdir -p /data/heirs-property/postgres/data

# Permissions
chown -R $USER:$USER /data/heirs-property
chmod -R 755 /data/heirs-property
```

### Phase 2: Data Migration
```bash
# Backup current data
tar czf /data/heirs-property-backup-$(date +%Y%m%d).tar.gz ./data

# Move existing data
rsync -av --progress ./data/ /data/heirs-property/

# Create symlink
rm -rf ./data
ln -s /data/heirs-property data
```

### Phase 3: Docker Configuration
```yaml
# docker-compose.yml update
volumes:
  postgres_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /data/heirs-property/postgres
```

## 4. Data Management Policies

### A. Retention Policies
| Data Type | Retention Period | Backup Frequency |
|-----------|-----------------|------------------|
| Raw Data | Permanent | Monthly |
| Processed Data | Last 3 versions | Weekly |
| Interim Data | 7 days | None |
| Database | Full history | Daily |

### B. Cleanup Procedures
```python
from pathlib import Path
from datetime import datetime, timedelta

def cleanup_interim_data():
    """Clean interim data older than 7 days."""
    interim_path = Path("/data/heirs-property/interim")
    cutoff = datetime.now() - timedelta(days=7)
    
    for file in interim_path.glob("**/*"):
        if file.stat().st_mtime < cutoff.timestamp():
            file.unlink()

def cleanup_processed_versions():
    """Keep only last 3 versions of processed data."""
    processed_path = Path("/data/heirs-property/processed")
    for category in ['parquet', 'vectors', 'rasters']:
        files = sorted(
            processed_path.glob(f"{category}/*"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        for old_file in files[3:]:
            old_file.unlink()
```

## 5. Monitoring System

### A. Storage Monitoring
```python
import psutil
import logging
from pathlib import Path

class StorageMonitor:
    """Monitor storage usage and alert on thresholds."""
    
    def __init__(self):
        self.thresholds = {
            '/data': 80,  # Alert at 80% usage
            '/': 90      # Alert at 90% usage
        }
        self.logger = logging.getLogger(__name__)
    
    def check_usage(self):
        """Check storage usage against thresholds."""
        for path, threshold in self.thresholds.items():
            usage = psutil.disk_usage(path)
            if usage.percent >= threshold:
                self.alert(path, usage.percent)
    
    def alert(self, path: str, usage: float):
        """Send alert for high storage usage."""
        message = f"Storage alert: {path} is at {usage}% usage"
        self.logger.warning(message)
        # Add additional alert mechanisms (email, Slack, etc.)
```

### B. Monitoring Metrics
| Metric | Threshold | Action |
|--------|-----------|--------|
| Root Volume Usage | 90% | Alert + Cleanup |
| Data Volume Usage | 80% | Alert |
| Database Size | 10GB | Optimize |
| Interim Data Age | 7 days | Auto-delete |

## 6. Backup Strategy

### A. Database Backups
```bash
#!/bin/bash
# backup_database.sh

BACKUP_DIR="/data/heirs-property/backup/postgres"
RETENTION_DAYS=5

# Create backup
pg_dump -h localhost -U heirs_user -d heirs_property > \
    "$BACKUP_DIR/backup_$(date +%Y%m%d).sql"

# Remove old backups
find $BACKUP_DIR -name "backup_*.sql" -mtime +$RETENTION_DAYS -delete
```

### B. Data Backups
```bash
#!/bin/bash
# backup_data.sh

BACKUP_DIR="/data/heirs-property/backup"
RAW_DATA="/data/heirs-property/raw"
PROCESSED_DATA="/data/heirs-property/processed"

# Backup raw data (monthly)
if [ $(date +%d) = "01" ]; then
    tar czf "$BACKUP_DIR/raw_$(date +%Y%m).tar.gz" $RAW_DATA
fi

# Backup processed data (weekly)
if [ $(date +%u) = "7" ]; then
    tar czf "$BACKUP_DIR/processed_$(date +%Y%m%d).tar.gz" $PROCESSED_DATA
fi
```

## 7. Success Criteria and Verification

### A. Storage Efficiency
- [ ] Root volume usage below 80%
- [ ] Data volume usage below 70%
- [ ] No temporary files older than 7 days
- [ ] Compressed storage formats used where appropriate

### B. Data Safety
- [ ] Daily database backups successful
- [ ] Weekly processed data backups complete
- [ ] Monthly raw data verification
- [ ] All critical paths monitored

### C. Performance
- [ ] Database query performance maintained
- [ ] Backup operations completed within maintenance window
- [ ] No storage-related processing delays

## 8. Maintenance Schedule

| Task | Frequency | Responsible |
|------|-----------|-------------|
| Cleanup Check | Daily | Automated |
| Usage Monitoring | Hourly | Automated |
| Backup Verification | Weekly | Manual |
| Performance Review | Monthly | Manual |

## 9. Emergency Procedures

### A. Storage Full Response
1. Stop non-critical processes
2. Run emergency cleanup
3. Verify backup integrity
4. Expand storage if needed

### B. Recovery Procedures
1. Identify data loss/corruption
2. Stop affected processes
3. Restore from latest backup
4. Verify data integrity
5. Resume operations

## 10. Future Considerations

### A. Scaling Strategy
- Monitor growth rates
- Plan storage expansions
- Consider cloud backup options
- Evaluate compression strategies

### B. Performance Optimization
- Regular database maintenance
- Storage access patterns analysis
- Caching strategy review
- I/O optimization

## 11. Documentation Updates

This plan should be reviewed and updated:
- Monthly during development
- After significant storage events
- When new data types are added
- When storage patterns change significantly
``` 