"""
ShortV module for VLM optimization via visual token skipping
"""

from .core import ShortV

# Import adapters to register them
from . import adapters

__all__ = ["ShortV"]