"""
Base monkey patch system for ShortX
"""

import torch
import types
from typing import Callable, Dict, List, Any
from enum import Enum

from .logger import logger


class PruningStrategy(Enum):
    """Available pruning strategies"""
    IDENTITY = "identity"           # Skip layer completely (identity function)
    ATTENTION_ONLY = "attn_only"    # Keep only attention, skip FFN
    FFN_ONLY = "ffn_only"          # Keep only FFN, skip attention
    LAST_ATTN_FFN_ONLY = "last_attn_ffn_only"  # Keep only last token for both attention and FFN


class BasePatcher:
    """
    Base class for model patching systems

    Provides common functionality for monkey patching model layers
    with different pruning strategies using independent strategy functions.
    """

    def __init__(self, model: torch.nn.Module, model_type: str = None):
        """
        Initialize BasePatcher

        Args:
            model: PyTorch model to patch
            model_type: Model type (auto-detected if not provided)
        """
        self.model = model
        self.model_type = model_type or self._detect_model_type()
        self._original_forwards: Dict[int, Callable] = {}
        self._active_patches: Dict[int, str] = {}

    def _detect_model_type(self) -> str:
        """Auto-detect model type from config"""
        if hasattr(self.model, 'config'):
            return self.model.config.model_type.lower()
        return "unknown"

    def _get_layers(self) -> torch.nn.ModuleList:
        """Get model layers - should be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _get_layers")

    def patch_layer(self, layer_idx: int, strategy_func: Callable, strategy_name: str = None):
        """
        Apply a strategy function to a specific layer

        Args:
            layer_idx: Index of layer to patch
            strategy_func: Function to use as new forward
            strategy_name: Name of the strategy for logging
        """
        layers = self._get_layers()

        if layer_idx >= len(layers):
            raise ValueError(f"Layer index {layer_idx} out of range (max: {len(layers)-1})")

        target_layer = layers[layer_idx]

        # Backup original forward if not already done
        if layer_idx not in self._original_forwards:
            self._original_forwards[layer_idx] = target_layer.forward

        # Apply new forward function
        new_forward = types.MethodType(strategy_func, target_layer)
        target_layer.forward = new_forward

        strategy_name = strategy_name or strategy_func.__name__
        self._active_patches[layer_idx] = strategy_name
        logger.success(f"✅ Applied {strategy_name} to layer {layer_idx}")

    def unpatch_layer(self, layer_idx: int):
        """Restore original forward function for a specific layer"""
        if layer_idx not in self._original_forwards:
            logger.warning(f"Layer {layer_idx} was not patched")
            return

        layers = self._get_layers()
        target_layer = layers[layer_idx]
        target_layer.forward = self._original_forwards[layer_idx]

        del self._original_forwards[layer_idx]
        if layer_idx in self._active_patches:
            del self._active_patches[layer_idx]

        logger.success(f"✅ Restored original forward for layer {layer_idx}")

    def unpatch_all(self):
        """Restore all layers to original state"""
        for layer_idx in list(self._original_forwards.keys()):
            self.unpatch_layer(layer_idx)

        logger.success("✅ All layers restored to original state")

    def get_patch_info(self) -> Dict[str, Any]:
        """Get information about current patches"""
        return {
            "model_type": self.model_type,
            "total_layers": len(self._get_layers()),
            "patched_layers": dict(self._active_patches),
            "num_patched": len(self._active_patches),
        }

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - restore all patches"""
        self.unpatch_all()