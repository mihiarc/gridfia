#!/bin/bash

# Data backup script
BACKUP_DIR="/data/heirs-property/backup"
RAW_DATA="/data/heirs-property/raw"
PROCESSED_DATA="/data/heirs-property/processed"
DATE=$(date +%Y%m%d)

# Ensure backup directories exist
mkdir -p "$BACKUP_DIR/raw"
mkdir -p "$BACKUP_DIR/processed"

# Backup raw data (monthly)
if [ $(date +%d) = "01" ]; then
    echo "Creating monthly raw data backup..."
    tar czf "$BACKUP_DIR/raw/raw_$(date +%Y%m).tar.gz" $RAW_DATA
fi

# Backup processed data (weekly)
if [ $(date +%u) = "7" ]; then
    echo "Creating weekly processed data backup..."
    tar czf "$BACKUP_DIR/processed/processed_$DATE.tar.gz" $PROCESSED_DATA
fi

# Clean up old processed backups (keep last 4 weeks)
find "$BACKUP_DIR/processed" -name "processed_*.tar.gz" -mtime +28 -delete

echo "Backup tasks complete." 