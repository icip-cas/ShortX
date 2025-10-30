"""
Logger configuration for ShortX
"""

from loguru import logger
import sys

# Configure loguru logger with custom format and colors
def configure_logger():
    """Configure loguru logger with custom formatting"""
    # Remove default handler
    logger.remove()
    
    # Add custom handler with emoji and colors
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        level="INFO",
        colorize=True,
        enqueue=True,
    )
    
    return logger

# Configure logger on import
configure_logger()

# Export logger instance
__all__ = ["logger"]