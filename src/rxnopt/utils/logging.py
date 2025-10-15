"""Modern logging configuration for rxnopt.

Provides structured logging with rich formatting and multiple output options.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install
from loguru import logger

# Install rich traceback handler
install(show_locals=True)

console = Console()


def configure_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    rich_tracebacks: bool = True,
) -> None:
    """Configure logging with rich formatting.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file to write logs to
        rich_tracebacks: Whether to use rich traceback formatting
    """
    # Remove default logger
    logger.remove()
    
    # Add rich console handler
    logger.add(
        RichHandler(console=console, rich_tracebacks=rich_tracebacks),
        format="{time:HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level,
    )
    
    # Add file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=level,
            rotation="10 MB",
            retention="1 week",
            compression="gz",
        )
        
        logger.info(f"Logging to file: {log_file}")


def get_logger(name: str = __name__) -> logger:
    """Get a configured logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    return logger.bind(name=name)


# Configure default logging
configure_logging()

# Export the default logger
__all__ = ["configure_logging", "get_logger", "logger", "console"]