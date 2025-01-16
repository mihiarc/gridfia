"""
Centralized error handling for the heirs-property project.
"""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ProcessingError(Exception):
    """Custom exception for processing errors."""
    message: str
    context: Dict[str, Any]
    timestamp: datetime = datetime.now()

@dataclass
class ValidationError(Exception):
    """Custom exception for data validation errors."""
    message: str
    data_type: str
    details: Dict[str, Any]
    timestamp: datetime = datetime.now()

class ErrorHandler:
    """Handles errors across the application with standardized logging and responses."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the error handler.
        
        Args:
            logger: Optional logger instance. If None, creates a new one.
        """
        self.logger = logger or logging.getLogger(__name__)
        
    def handle_processing_error(self, error: Exception, context: str) -> ProcessingError:
        """
        Handle processing-related errors with context.
        
        Args:
            error: The original exception
            context: Description of what was being processed
            
        Returns:
            ProcessingError: A standardized processing error
        """
        error_context = {
            'original_error': str(error),
            'error_type': error.__class__.__name__,
            'processing_context': context
        }
        
        self.logger.error(f"Processing error in {context}: {str(error)}", 
                         extra={'error_context': error_context})
        
        return ProcessingError(
            message=f"Error processing {context}: {str(error)}",
            context=error_context
        )
        
    def handle_validation_error(self, error: Exception, data_type: str,
                              details: Dict[str, Any]) -> ValidationError:
        """
        Handle data validation errors.
        
        Args:
            error: The original exception
            data_type: Type of data being validated
            details: Additional validation context
            
        Returns:
            ValidationError: A standardized validation error
        """
        self.logger.error(f"Validation error for {data_type}: {str(error)}",
                         extra={'validation_details': details})
        
        return ValidationError(
            message=str(error),
            data_type=data_type,
            details=details
        ) 