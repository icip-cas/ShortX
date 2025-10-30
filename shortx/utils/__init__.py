"""
Utility functions for ShortX
"""

import os
import json
from typing import Dict, Any, List, Optional

# Import patch utilities
from .patch import BasePatcher, PruningStrategy
from .logger import logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to JSON file"""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def get_model_type(model_name: str) -> str:
    """Determine model type from name"""
    model_name_lower = model_name.lower()
    
    if any(x in model_name_lower for x in ["llava", "vila", "video"]):
        return "vlm"
    elif any(x in model_name_lower for x in ["llama", "baichuan", "mistral", "qwen", "gpt"]):
        return "llm"
    else:
        return "unknown"


def format_layer_list(layers: List[int]) -> str:
    """Format layer list for display"""
    if not layers:
        return "None"
    
    # Group consecutive layers
    groups = []
    current_group = [layers[0]]
    
    for layer in layers[1:]:
        if layer == current_group[-1] + 1:
            current_group.append(layer)
        else:
            groups.append(current_group)
            current_group = [layer]
    
    groups.append(current_group)
    
    # Format groups
    formatted = []
    for group in groups:
        if len(group) == 1:
            formatted.append(str(group[0]))
        else:
            formatted.append(f"{group[0]}-{group[-1]}")
    
    return ", ".join(formatted)


def parse_layer_string(layer_string: str) -> List[int]:
    """Parse layer string into list of integers"""
    layers = []
    
    for part in layer_string.split(","):
        part = part.strip()
        if "-" in part:
            # Range of layers
            start, end = part.split("-")
            layers.extend(range(int(start), int(end) + 1))
        else:
            # Single layer
            layers.append(int(part))
    
    return sorted(list(set(layers)))


def get_cache_dir() -> str:
    """Get cache directory for ShortX"""
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "shortx")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def download_file(url: str, dest_path: str, progress: bool = True):
    """Download file with progress bar"""
    import requests
    from tqdm import tqdm
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f:
        if progress and total_size > 0:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(dest_path)) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        else:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)


def check_dependencies(module_type: str) -> Dict[str, bool]:
    """Check if required dependencies are installed"""
    dependencies = {
        "core": ["torch", "transformers", "numpy"],
        "shortgpt": ["datasets", "tqdm"],
        "shortv": ["pillow", "gradio", "einops"],
        "eval": ["lmms_eval", "accelerate"]
    }
    
    results = {}
    for dep in dependencies.get(module_type, []):
        try:
            __import__(dep)
            results[dep] = True
        except ImportError:
            results[dep] = False
    
    return results


def print_system_info():
    """Print system information for debugging"""
    import platform
    import torch
    
    logger.info("System Information:")
    logger.info(f"  Python: {platform.python_version()}")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        logger.info(f"  CUDA: {torch.version.cuda}")
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.info("  CUDA: Not available")


class ProgressTracker:
    """Track progress for long-running operations"""
    
    def __init__(self, total_steps: int, description: str = "Progress"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        
        try:
            from tqdm import tqdm
            self.pbar = tqdm(total=total_steps, desc=description)
        except ImportError:
            self.pbar = None
    
    def update(self, n: int = 1):
        """Update progress"""
        self.current_step += n
        if self.pbar:
            self.pbar.update(n)
        else:
            logger.info(f"{self.description}: {self.current_step}/{self.total_steps}")
    
    def close(self):
        """Close progress bar"""
        if self.pbar:
            self.pbar.close()