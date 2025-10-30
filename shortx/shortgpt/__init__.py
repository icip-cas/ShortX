"""
ShortGPT module for layer pruning in LLMs
"""

from .core import ShortGPT

# Import adapters to register them
from . import adapters

__all__ = ["ShortGPT"]