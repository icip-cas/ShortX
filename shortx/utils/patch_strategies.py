"""
Common patch strategy functions for monkey patching

This file contains only the universal strategies that can be used across all models.
Model-specific strategies are defined in their respective adapter files.
"""

def identity_forward(layer, hidden_states, *args, **kwargs):
    """Identity forward - skip layer completely
    
    Args:
        layer: The layer object (bound as self when used as method)
        hidden_states: Input hidden states tensor
        
    Returns:
        hidden_states unchanged (skip layer processing)
    """
    return (hidden_states,)