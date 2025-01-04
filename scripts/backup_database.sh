#!/bin/bash

# Database backup script
BACKUP_DIR="/data/heirs-property/backup/postgres"
RETENTION_DAYS=5
DATE=$(date +%Y%m%d)

# Ensure backup directory exists
mkdir -p $BACKUP_DIR

# Create backup
echo "Creating database backup for $DATE..."
docker-compose exec -T postgis pg_dump -U heirs_user -d heirs_property > \
    "$BACKUP_DIR/backup_$DATE.sql"

# Remove old backups
echo "Cleaning up old backups..."
find $BACKUP_DIR -name "backup_*.sql" -mtime +$RETENTION_DAYS -delete

echo "Backup complete." 