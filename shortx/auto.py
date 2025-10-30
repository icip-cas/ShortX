"""
AutoModel system for ShortX - provides automatic model mapping similar to transformers AutoModel
"""

from typing import Dict, Type, Optional, Union
from transformers import AutoConfig
from .utils.logger import logger


class ModelRegistry:
    """Central registry for model adapters"""
    
    _shortgpt_mapping: Dict[str, Type] = {}
    _shortv_mapping: Dict[str, Type] = {}
    
    @classmethod
    def register_shortgpt(cls, model_type: str, adapter_class: Type):
        """Register a ShortGPT adapter for a model type"""
        cls._shortgpt_mapping[model_type] = adapter_class
    
    @classmethod
    def register_shortv(cls, model_type: str, adapter_class: Type):
        """Register a ShortV adapter for a model type"""
        cls._shortv_mapping[model_type] = adapter_class
    
    @classmethod
    def get_shortgpt_adapter(cls, model_type: str) -> Optional[Type]:
        """Get ShortGPT adapter for model type"""
        return cls._shortgpt_mapping.get(model_type)
    
    @classmethod
    def get_shortv_adapter(cls, model_type: str) -> Optional[Type]:
        """Get ShortV adapter for model type"""
        return cls._shortv_mapping.get(model_type)


class AutoShortGPT:
    """Automatically create ShortGPT instance based on model type"""
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        """
        Create ShortGPT instance automatically based on model type
        
        Args:
            model_name_or_path: Model name or local path
            **kwargs: Additional arguments for the model adapter
            
        Returns:
            ShortGPT instance with appropriate adapter
        """
        # Get model config to determine model type
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        model_type = config.model_type.lower()
        
        # Get the appropriate adapter
        adapter_class = ModelRegistry.get_shortgpt_adapter(model_type)
        
        if adapter_class is None:
            # Fall back to base ShortGPT
            from .shortgpt.core import ShortGPT
            logger.warning(f"No specific adapter found for model type '{model_type}', using base ShortGPT")
            return ShortGPT(model_name_or_path, **kwargs)
        
        logger.info(f"Using {adapter_class.__name__} for model type '{model_type}'")
        return adapter_class(model_name_or_path, **kwargs)


class AutoShortV:
    """Automatically create ShortV instance based on model type"""
    
    @classmethod 
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        """
        Create ShortV instance automatically based on model type
        
        Args:
            model_name_or_path: Model name or local path
            **kwargs: Additional arguments for the model adapter
            
        Returns:
            ShortV instance with appropriate adapter
        """
        # Get model config to determine model type
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        model_type = config.model_type.lower()
        
        # Get the appropriate adapter
        adapter_class = ModelRegistry.get_shortv_adapter(model_type)
        
        if adapter_class is None:
            # Fall back to base ShortV
            from .shortv.core import ShortV
            logger.warning(f"No specific adapter found for model type '{model_type}', using base ShortV")
            return ShortV(model_name_or_path, **kwargs)
        
        logger.info(f"Using {adapter_class.__name__} for model type '{model_type}'")
        return adapter_class(model_name_or_path, **kwargs)


def list_supported_models() -> Dict[str, Dict[str, list]]:
    """List all supported model types and their adapters"""
    return {
        "shortgpt": {
            "supported_types": list(ModelRegistry._shortgpt_mapping.keys()),
            "adapters": [cls.__name__ for cls in ModelRegistry._shortgpt_mapping.values()]
        },
        "shortv": {
            "supported_types": list(ModelRegistry._shortv_mapping.keys()),
            "adapters": [cls.__name__ for cls in ModelRegistry._shortv_mapping.values()]
        }
    }