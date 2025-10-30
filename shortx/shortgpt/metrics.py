"""
Layer importance metrics for ShortGPT

This module provides three core metrics for evaluating layer importance
in transformer models: Block Influence, L2 Distance, and Variance.
"""

import torch
from typing import Dict, Callable

from ..utils.logger import logger


def block_influence(in_hidden: torch.Tensor, out_hidden: torch.Tensor) -> torch.Tensor:
    """
    Block Influence metric - measures layer contribution via cosine similarity
    
    Originally proposed in the ShortGPT paper. Calculates 1 - cosine_similarity
    between input and output hidden states to measure layer contribution.
    
    Args:
        in_hidden: Input hidden states to the layer
        out_hidden: Output hidden states from the layer
        
    Returns:
        Block influence scores
    """
    _, _, d = in_hidden.shape
    in_flat = in_hidden.reshape(-1, d)
    out_flat = out_hidden.reshape(-1, d)

    norm_in = in_flat.norm(dim=-1, keepdim=True)
    norm_out = out_flat.norm(dim=-1, keepdim=True)

    sim = (in_flat @ out_flat.T) / (norm_in * norm_out)
    sim = sim.diagonal().nan_to_num(nan=0.5)
    
    return 1 - sim


def l2_distance(in_hidden: torch.Tensor, out_hidden: torch.Tensor) -> torch.Tensor:
    """
    L2 Distance metric - measures L2 norm of changes between layers
    
    Computes the L2 norm of the difference between input and output hidden states.
    Higher values indicate more significant transformations by the layer.
    
    Args:
        in_hidden: Input hidden states to the layer
        out_hidden: Output hidden states from the layer
        
    Returns:
        L2 distance scores
    """
    diff = out_hidden - in_hidden
    _, _, d = diff.shape
    diff_flat = diff.reshape(-1, d)
    l2_scores = torch.norm(diff_flat, dim=1, p=2)
    
    return l2_scores


def variance_change(in_hidden: torch.Tensor, out_hidden: torch.Tensor) -> torch.Tensor:
    """
    Variance Change metric - measures variance changes between layers
    
    Computes the absolute difference in variance between input and output
    hidden states to measure the layer's impact on representation diversity.
    
    Args:
        in_hidden: Input hidden states to the layer
        out_hidden: Output hidden states from the layer
        
    Returns:
        Variance change scores
    """
    _, _, d = in_hidden.shape
    in_flat = in_hidden.reshape(-1, d)
    out_flat = out_hidden.reshape(-1, d)
    
    # Calculate variance along feature dimension
    in_var = torch.var(in_flat, dim=1)
    out_var = torch.var(out_flat, dim=1)
    
    # Use absolute difference in variance as importance measure
    var_change = torch.abs(out_var - in_var)
    
    return var_change


# Registry of available metrics
METRICS_REGISTRY: Dict[str, Callable] = {
    "block_influence": block_influence,
    "l2_distance": l2_distance, 
    "variance_change": variance_change,
}


def get_metric_function(metric_name: str) -> Callable:
    """
    Get metric function by name
    
    Args:
        metric_name: Name of the metric to retrieve
        
    Returns:
        Metric function
        
    Raises:
        ValueError: If metric name is not found in registry
    """
    if metric_name not in METRICS_REGISTRY:
        available_metrics = list(METRICS_REGISTRY.keys())
        raise ValueError(
            f"Unknown metric '{metric_name}'. Available metrics: {available_metrics}"
        )
    
    return METRICS_REGISTRY[metric_name]


def list_available_metrics() -> Dict[str, str]:
    """
    List all available metrics with descriptions
    
    Returns:
        Dictionary mapping metric names to descriptions
    """
    descriptions = {
        "block_influence": "Block Influence - measures layer contribution via cosine similarity",
        "l2_distance": "L2 Distance - L2 norm of hidden state changes", 
        "variance_change": "Variance Change - measures variance changes between layers",
    }
    
    return {name: descriptions.get(name, "Custom metric") for name in METRICS_REGISTRY.keys()}