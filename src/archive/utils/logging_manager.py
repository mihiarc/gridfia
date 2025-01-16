"""
Centralized logging configuration for the heirs-property project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

class LoggingManager:
    """Manages logging configuration across the application."""
    
    def __init__(self, log_level: str = "INFO", log_file: Optional[Path] = None):
        """
        Initialize the logging manager.
        
        Args:
            log_level: The logging level to use (default: "INFO")
            log_file: Optional path to a log file. If None, logs to stdout only.
        """
        self.log_level = getattr(logging, log_level.upper())
        self.log_file = log_file
        
    def setup_logging(self, name: str) -> logging.Logger:
        """
        Set up a logger with standardized configuration.
        
        Args:
            name: The name for the logger, typically __name__ of the calling module
            
        Returns:
            A configured logger instance
        """
        logger = logging.getLogger(name)
        logger.setLevel(self.log_level)
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if log_file is specified
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
        return logger 