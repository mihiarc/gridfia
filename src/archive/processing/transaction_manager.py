"""
Transaction management module for database operations.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import contextmanager

from sqlalchemy import create_engine, text

class TransactionManager:
    """Manages database transactions with checkpointing."""
    
    def __init__(
        self,
        db_connection: str,
        checkpoint_dir: Path,
        schema: str = "public"
    ):
        """Initialize transaction manager.
        
        Args:
            db_connection: Database connection string
            checkpoint_dir: Directory for checkpoint files
            schema: Database schema (default: public)
        """
        self.engine = create_engine(db_connection)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.schema = schema
        self.active_transactions = {}
    
    def _execute_statements(self, conn, statements: list[str]):
        """Execute multiple SQL statements sequentially.
        
        Args:
            conn: Database connection
            statements: List of SQL statements to execute
        """
        for stmt in statements:
            if stmt.strip():  # Skip empty statements
                conn.execute(text(stmt))
                conn.commit()
    
    @contextmanager
    def transaction(self, table_name: str):
        """Create a transaction context.
        
        Args:
            table_name: Name of the table being modified
            
        Yields:
            Database engine for the transaction
        """
        # Create transaction ID using a safe format for table names
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        transaction_id = f"{table_name}_{timestamp}"
        backup_table = f"{table_name}_backup_{timestamp}"
        
        try:
            # Start transaction
            self.active_transactions[transaction_id] = {
                "table_name": table_name,
                "schema": self.schema,
                "start_time": datetime.now().isoformat(),
                "state": "active"
            }
            
            # Create backup table
            with self.engine.connect() as conn:
                self._execute_statements(conn, [
                    f"CREATE TABLE IF NOT EXISTS {backup_table} AS SELECT * FROM {table_name}"
                ])
            
            # Yield engine for operations
            yield self.engine
            
            # Cleanup on success
            with self.engine.connect() as conn:
                self._execute_statements(conn, [
                    f"DROP TABLE IF EXISTS {backup_table}"
                ])
            
            # Update transaction state
            self.active_transactions[transaction_id]["state"] = "completed"
            self._save_checkpoint(transaction_id)
            
        except Exception as e:
            # Rollback on failure
            with self.engine.connect() as conn:
                self._execute_statements(conn, [
                    f"DROP TABLE IF EXISTS {table_name}",
                    f"CREATE TABLE {table_name} AS SELECT * FROM {backup_table}",
                    f"DROP TABLE IF EXISTS {backup_table}"
                ])
            
            # Update transaction state
            self.active_transactions[transaction_id]["state"] = "failed"
            self.active_transactions[transaction_id]["error"] = str(e)
            self.active_transactions[transaction_id]["error_time"] = (
                datetime.now().isoformat()
            )
            self._save_checkpoint(transaction_id)
            
            raise
        
        finally:
            # Cleanup transaction
            self.active_transactions.pop(transaction_id, None)
    
    def _save_checkpoint(self, transaction_id: str):
        """Save transaction checkpoint.
        
        Args:
            transaction_id: ID of the transaction
        """
        checkpoint_path = (
            self.checkpoint_dir / f"checkpoint_{transaction_id}.json"
        )
        
        checkpoint = {
            "transaction_id": transaction_id,
            "table_name": self.active_transactions[transaction_id]["table_name"],
            "schema": self.schema,
            "timestamp": datetime.now().isoformat(),
            "state": self.active_transactions[transaction_id]
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def list_active_transactions(self) -> list[Dict[str, Any]]:
        """List all active transactions.
        
        Returns:
            List of active transaction details
        """
        return [
            {
                "transaction_id": tid,
                "table_name": details["table_name"],
                "schema": details["schema"],
                "start_time": details["start_time"],
                "state": details["state"]
            }
            for tid, details in self.active_transactions.items()
        ]
    
    def get_transaction_status(
        self,
        transaction_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get status of a specific transaction.
        
        Args:
            transaction_id: ID of the transaction
            
        Returns:
            Transaction details or None if not found
        """
        return self.active_transactions.get(transaction_id)
    
    def load_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Load a transaction checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint data
        """
        with open(checkpoint_path) as f:
            return json.load(f)
    
    def restore_from_checkpoint(
        self,
        checkpoint_path: Path
    ) -> Optional[Dict[str, Any]]:
        """Restore database state from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Restoration status
        """
        try:
            checkpoint = self.load_checkpoint(checkpoint_path)
            table_name = checkpoint["table_name"]
            backup_table = f"{table_name}_backup_{checkpoint['transaction_id']}"
            
            with self.engine.connect() as conn:
                # Check if backup exists
                result = conn.execute(text(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = :backup_table)"
                ), {"backup_table": backup_table})
                
                if result.scalar():
                    # Restore from backup
                    self._execute_statements(conn, [
                        f"DROP TABLE IF EXISTS {table_name}",
                        f"CREATE TABLE {table_name} AS SELECT * FROM {backup_table}",
                        f"DROP TABLE IF EXISTS {backup_table}"
                    ])
                    
                    return {
                        "status": "restored",
                        "table_name": table_name,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {
                        "status": "no_backup",
                        "table_name": table_name,
                        "timestamp": datetime.now().isoformat()
                    }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            } 