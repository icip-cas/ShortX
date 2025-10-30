"""
Core ShortV implementation - Defines the abstract interface for VLM optimization.
"""
from typing import List, Optional, Dict, Any, Callable
from abc import ABC, abstractmethod

from ..utils.logger import logger


class ShortV(ABC):
    """
    ShortV: Abstract base class defining the interface for VLM optimization 
    by selectively skipping visual token computation.
    
    All implementation logic is delegated to specific adapter classes (e.g., LLaVAShortV).
    """
    
    def __init__(self, model_name: str, n_skip_layers: Optional[int] = None,
                 skip_layers: Optional[List[int]] = None,
                 device_map: str = "auto", torch_dtype: str = "auto"):
        """
        Initializes ShortV.
        
        Args:
            model_name: The HuggingFace model name.
            n_skip_layers: The number of layers to skip (for automatic selection).
            skip_layers: The indices of layers to skip explicitly.
            device_map: The device mapping strategy.
            torch_dtype: The PyTorch data type.
        """
        self.model_name = model_name
        self.n_skip_layers = n_skip_layers
        self.skip_layers = skip_layers or []
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        
        self.model = None
        self.processor = None
        self.layers_importance = None
        self._patching_enabled = False

    @abstractmethod
    def enable_monkey_patching(self):
        """
        Enables model patching for visual token skipping.
        This method must be implemented by the adapter.
        """
        pass

    @abstractmethod
    def disable_monkey_patching(self):
        """
        Disables model patching and restores the original model state.
        This method must be implemented by the adapter.
        """
        pass

    @abstractmethod
    def set_skip_layers(self, layers: List[int]):
        """
        Sets the layers to be replaced/skipped and handles model reloading.
        This method must be implemented by the adapter.
        """
        pass

    @abstractmethod
    def calculate_lc_scores(self, evaluation_function: Callable[..., Any], **kwargs) -> Dict[str, Any]:
        """
        Calculates Layer Contribution (LC) scores by running an evaluation.
        This method must be implemented by the adapter.
        """
        pass