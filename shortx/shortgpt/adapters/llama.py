"""
Llama model adapter for ShortGPT with integrated patcher
"""

from typing import List, Optional, Dict, Any
import torch
from ..core import ShortGPT
from ...utils.patch import BasePatcher, PruningStrategy
from ...utils.patch_strategies import identity_forward
from ...auto import ModelRegistry
from ...utils.logger import logger


# ================================
# Llama-specific patch strategies
# ================================

def llama_attention_only_forward(
    layer,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    use_cache=None,
    cache_position=None,
    position_embeddings=None,
    *args,
    **kwargs
):
    """Llama-specific attention only forward
    
    Args:
        layer: Llama layer object
        hidden_states: Input hidden states
        
    Returns:
        Hidden states after attention processing only
    """
    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)
    
    # Self Attention
    hidden_states, _ = layer.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + hidden_states
    return (hidden_states,)


def llama_ffn_only_forward(layer, hidden_states, *args, **kwargs):
    """Llama-specific FFN only forward
    
    Args:
        layer: Llama layer object  
        hidden_states: Input hidden states
        
    Returns:
        Hidden states after FFN processing only
    """
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return (hidden_states,)


def llama_last_attn_ffn_only_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    use_cache=None,
    cache_position=None,
    position_embeddings=None,
    *args,
    **kwargs
):
    """Llama-specific last token attention+FFN only forward
    
    Args:
        self: Llama layer object (bound method)
        hidden_states: Input hidden states
        
    Returns:
        Hidden states with only last token processed through both attention and FFN
    """
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    
    # Self Attention
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states[..., :-1, :] = 0
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states[..., :-1, :] = 0
    hidden_states = residual + hidden_states
    return (hidden_states,)


class LlamaPatcher(BasePatcher):
    """Patcher specifically designed for Llama models"""
    
    def _get_layers(self) -> torch.nn.ModuleList:
        """Get Llama model layers"""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        
        raise ValueError(f"Could not find layers in {self.model_type} model")
    
    def apply_identity_pruning(self, layer_indices):
        """Apply identity pruning to specified layers"""
        for idx in layer_indices:
            self.patch_layer(idx, identity_forward, "identity")
    
    def apply_attention_only_pruning(self, layer_indices):
        """Apply attention-only pruning to specified layers"""
        for idx in layer_indices:
            self.patch_layer(idx, llama_attention_only_forward, "llama_attention_only")
    
    def apply_ffn_only_pruning(self, layer_indices):
        """Apply FFN-only pruning to specified layers"""
        for idx in layer_indices:
            self.patch_layer(idx, llama_ffn_only_forward, "llama_ffn_only")
    
    def apply_last_attn_ffn_only_pruning(self, layer_indices):
        """Apply last token attention+FFN only pruning to specified layers"""
        for idx in layer_indices:
            self.patch_layer(idx, llama_last_attn_ffn_only_forward, "llama_last_attn_ffn_only")


class LlamaShortGPT(ShortGPT):
    """
    Llama-specific ShortGPT implementation with monkey patch support
    """
    
    def __init__(self, model_name: str, **kwargs):
        """Initialize Llama ShortGPT"""
        # Initialize base class with Llama-specific layers path  
        super().__init__(model_name, layers_path="model.layers", **kwargs)
        self.patcher = None
    
    def _get_layers_path(self, model_name: str) -> str:
        """Get Llama-specific layers path"""
        return "model.layers"
    
    def enable_monkey_patching(self) -> 'LlamaShortGPT':
        """Enable monkey patch mode for non-invasive pruning"""
        if self.patcher is None:
            self.patcher = LlamaPatcher(self.model, "llama")
        logger.success("✅ Monkey patching enabled for Llama model")
        return self
    
    def disable_monkey_patching(self) -> 'LlamaShortGPT':
        """Disable monkey patch mode and restore original layers"""
        if self.patcher is not None:
            self.patcher.unpatch_all()
            self.patcher = None
        logger.success("✅ Monkey patching disabled, model restored to original state")
        return self
    
    def prune_layers_patch(self, layers_to_prune: List[int], 
                          strategy: PruningStrategy = PruningStrategy.IDENTITY) -> 'LlamaShortGPT':
        """
        Prune layers using monkey patching (non-invasive)
        
        Args:
            layers_to_prune: List of layer indices to prune
            strategy: Pruning strategy to apply
            
        Returns:
            Self for method chaining
        """
        if self.patcher is None:
            self.enable_monkey_patching()
        
        if strategy == PruningStrategy.IDENTITY:
            self.patcher.apply_identity_pruning(layers_to_prune)
        elif strategy == PruningStrategy.ATTENTION_ONLY:
            self.patcher.apply_attention_only_pruning(layers_to_prune)
        elif strategy == PruningStrategy.FFN_ONLY:
            self.patcher.apply_ffn_only_pruning(layers_to_prune)
        elif strategy == PruningStrategy.LAST_ATTN_FFN_ONLY:
            self.patcher.apply_last_attn_ffn_only_pruning(layers_to_prune)
        
        logger.success(f"✅ Applied {strategy.value} pruning to {len(layers_to_prune)} layers: {layers_to_prune}")
        return self
    
    def prune_layers_mixed(self, layer_configs: Dict[int, PruningStrategy]) -> 'LlamaShortGPT':
        """
        Apply different pruning strategies to different layers
        
        Args:
            layer_configs: Dict mapping layer indices to pruning strategies
            
        Returns:
            Self for method chaining
        """
        if self.patcher is None:
            self.enable_monkey_patching()
        
        # Group layers by strategy for efficient application
        identity_layers = []
        attn_only_layers = []
        ffn_only_layers = []
        last_attn_ffn_only_layers = []
        
        for layer_idx, strategy in layer_configs.items():
            if strategy == PruningStrategy.IDENTITY:
                identity_layers.append(layer_idx)
            elif strategy == PruningStrategy.ATTENTION_ONLY:
                attn_only_layers.append(layer_idx)
            elif strategy == PruningStrategy.FFN_ONLY:
                ffn_only_layers.append(layer_idx)
            elif strategy == PruningStrategy.LAST_ATTN_FFN_ONLY:
                last_attn_ffn_only_layers.append(layer_idx)
        
        # Apply strategies in groups
        if identity_layers:
            self.patcher.apply_identity_pruning(identity_layers)
        if attn_only_layers:
            self.patcher.apply_attention_only_pruning(attn_only_layers)
        if ffn_only_layers:
            self.patcher.apply_ffn_only_pruning(ffn_only_layers)
        if last_attn_ffn_only_layers:
            self.patcher.apply_last_attn_ffn_only_pruning(last_attn_ffn_only_layers)
        
        strategies_summary = {}
        for strategy in layer_configs.values():
            if strategy not in strategies_summary:
                strategies_summary[strategy] = 0
            strategies_summary[strategy] += 1
        
        logger.success(f"✅ Applied mixed pruning strategies: {strategies_summary}")
        return self
    
    def get_patch_status(self) -> Dict[str, Any]:
        """Get current monkey patch status"""
        if self.patcher is None:
            return {"patching_enabled": False, "active_patches": {}}
        
        return {
            "patching_enabled": self.patcher is not None,
            **(self.patcher.get_patch_info() if self.patcher else {"active_patches": {}})
        }
    
    def __enter__(self):
        """Context manager support"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up patches when exiting context"""
        if self.patcher is not None:
            self.patcher.unpatch_all()


# Register Llama adapter
ModelRegistry.register_shortgpt("llama", LlamaShortGPT)