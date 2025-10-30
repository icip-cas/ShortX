"""
Model adapters for ShortGPT

Each adapter file contains both the patcher and the ShortGPT implementation
for a specific model architecture. To add a new model:

1. Create a new file (e.g., mistral.py)  
2. Implement MistralPatcher(BasePatcher) with model-specific forward functions
3. Implement MistralShortGPT(ShortGPT) with pruning methods
4. Register with ModelRegistry.register_shortgpt("mistral", MistralShortGPT)
"""

from .qwen import QwenShortGPT
from .llama import LlamaShortGPT

__all__ = ["QwenShortGPT", "LlamaShortGPT"]