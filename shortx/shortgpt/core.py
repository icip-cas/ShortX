"""
Core ShortGPT implementation - Base class for model adapters
"""


import os
import re
import json
import random
import torch
import numpy as np
from typing import List, Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..utils.logger import logger


class ShortGPT:
    """
    Base ShortGPT class: Identifies and removes redundant layers in LLMs

    This is a base class that should be subclassed by model-specific adapters
    to handle different model architectures properly.
    """

    def __init__(self, model_name: str, layers_path: str = "model.layers",
                 n_prune_layers: Optional[int] = None, device_map: str = "auto",
                 torch_dtype: str = "auto"):
        """
        Initialize ShortGPT base class

        Args:
            model_name: HuggingFace model name or path
            layers_path: Path to layers in model (e.g., "model.layers")
            n_prune_layers: Number of layers to prune
            device_map: Device mapping strategy for model loading ("auto", "cpu", etc.)
            torch_dtype: PyTorch data type ("auto", "float16", "float32", etc.)
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype
        )

        # Set up layers access
        modules = layers_path.split(".")
        mod = self.model
        for m in modules:
            mod = getattr(mod, m)
        self.layers = mod

        self.n_prune_layers = n_prune_layers
        self.importances = [0 for _ in self.layers]
        self.layers_path = layers_path

    def analyze_layers(self, dataset_samples: List[str] = None,
                      metric: str = "block_influence", **kwargs) -> Dict[str, Any]:
        """
        Analyze layer importance using specified metric

        Args:
            dataset_samples: Text samples for evaluation
            metric: Metric to use for importance calculation ("block_influence", "gradient", etc.)
            **kwargs: Additional arguments for the metric

        Returns:
            Dictionary containing analysis results
        """
        if dataset_samples is None:
            # Use default samples if none provided
            dataset_samples = self._get_default_samples()

        self.eval_importance(dataset_samples, metric=metric, **kwargs)

        # Sort layers by importance
        layer_ranking = np.argsort(np.array(self.importances)).tolist()

        return {
            "model": self.model_name,
            "total_layers": len(self.layers),
            "importances": self.importances,
            "layer_ranking": layer_ranking,
            "metric_used": metric,
            "recommended_prune_layers": layer_ranking[:self.n_prune_layers] if self.n_prune_layers else layer_ranking[:len(self.layers)//4]
        }

    def _get_default_samples(self) -> List[str]:
        """Get default text samples for evaluation from pg19 dataset"""

        # Path to the pg19 dataset
        data_path = os.path.join(os.path.dirname(__file__), "data", "pg19.jsonl")


        try:
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"pg19.jsonl not found at {data_path}. Please ensure the data file is available.")

            all_samples = []

            # Define target length ranges for diverse samples
            length_ranges = [
                (20, 100),     # Short snippets
                (100, 300),    # Medium sentences
                (300, 800),    # Long paragraphs
                (800, 2000),   # Extended passages
                (2000, 4000),  # Long passages
            ]

            samples_per_range = 20  # 20 samples per length range = 100 total

            with open(data_path, 'r', encoding='utf-8') as f:
                # Read more entries to get sufficient diversity
                raw_texts = []
                for i, line in enumerate(f):
                    if i >= 50:  # Read from more books for diversity
                        break
                    try:
                        data = json.loads(line.strip())
                        text = data.get('text', '')
                        if text and len(text) > 1000:  # Only use substantial texts
                            # Clean the text: remove excessive whitespace and formatting
                            cleaned_text = re.sub(r'\s+', ' ', text).strip()
                            raw_texts.append(cleaned_text)
                    except json.JSONDecodeError:
                        continue

            if not raw_texts:
                raise ValueError("No valid texts found in pg19.jsonl. Please check the data file format.")

            # Extract samples for each length range
            random.seed(42)  # For reproducibility

            for min_len, max_len in length_ranges:
                range_samples = []
                attempts = 0
                max_attempts = len(raw_texts) * 50  # Prevent infinite loops

                while len(range_samples) < samples_per_range and attempts < max_attempts:
                    attempts += 1

                    # Select random text
                    text = random.choice(raw_texts)

                    # Find a random starting position
                    if len(text) <= max_len:
                        sample = text
                    else:
                        start_pos = random.randint(0, len(text) - max_len)
                        sample = text[start_pos:start_pos + max_len]

                    # Try to end at a sentence boundary for better quality
                    if len(sample) > min_len:
                        # Look for sentence endings near the target length
                        for end_pos in range(min(len(sample), max_len), min_len - 1, -1):
                            if end_pos < len(sample) and sample[end_pos] in '.!?':
                                sample = sample[:end_pos + 1].strip()
                                break

                    # Ensure sample is within length range and has good quality
                    if (min_len <= len(sample) <= max_len and
                        len(sample.split()) >= 5 and  # At least 5 words
                        not sample.startswith(('***', '---', 'CHAPTER', 'INDEX')) and  # Skip headers
                        sample.count('.') >= 1):  # Has at least one sentence

                        range_samples.append(sample.strip())

                # If we didn't get enough samples, generate more from existing raw_texts
                while len(range_samples) < samples_per_range and raw_texts:
                    text = random.choice(raw_texts)
                    # Create sample of target length by taking substring
                    if len(text) > max_len:
                        start = random.randint(0, len(text) - max_len)
                        sample = text[start:start + random.randint(min_len, max_len)]
                    else:
                        sample = text[:random.randint(min_len, min(max_len, len(text)))]

                    if len(sample) >= min_len:
                        range_samples.append(sample.strip())

                all_samples.extend(range_samples[:samples_per_range])

            # Shuffle final samples for good distribution
            random.shuffle(all_samples)

            logger.success(f"✅ Loaded {len(all_samples)} diverse text samples from pg19.jsonl")
            logger.info(f"   Length distribution: {[(min_len, max_len, samples_per_range) for min_len, max_len in length_ranges]}")

            return all_samples[:100]  # Ensure exactly 100 samples

        except Exception as e:
            logger.error(f"Error reading pg19.jsonl: {e}")
            raise RuntimeError(f"Failed to load text samples from pg19.jsonl: {e}")

    def prune_layers(self, layers_to_remove: Optional[List[int]] = None) -> List[int]:
        """
        Remove specified layers from the model

        Args:
            layers_to_remove: List of layer indices to remove. If None, uses importance scores.

        Returns:
            List of removed layer indices
        """
        if layers_to_remove is None:
            if self.n_prune_layers is None:
                raise ValueError("Either specify layers_to_remove or set n_prune_layers")
            layers_to_remove = np.argsort(np.array(self.importances))[:self.n_prune_layers].tolist()

        # Store original indices before deletion
        for i, layer in enumerate(self.layers):
            if not hasattr(layer, '_original_index'):
                layer._original_index = i

        # Remove layers in reverse order to maintain indices
        for layer_idx in sorted(layers_to_remove, reverse=True):
            del self.layers[layer_idx]

        # Update model configuration to reflect the new layer count
        self._update_model_config(layers_to_remove)

        return layers_to_remove

    def _update_model_config(self, removed_layers: List[int]):
        """
        Update model configuration after layer pruning

        Args:
            removed_layers: List of removed layer indices
        """
        config = self.model.config

        # Update num_hidden_layers
        original_layers = config.num_hidden_layers
        new_num_layers = original_layers - len(removed_layers)
        config.num_hidden_layers = new_num_layers

        # Update layer_types if it exists (for models like Qwen2)
        if hasattr(config, 'layer_types') and config.layer_types:
            # Remove corresponding entries from layer_types
            new_layer_types = []
            for i, layer_type in enumerate(config.layer_types):
                if i not in removed_layers:
                    new_layer_types.append(layer_type)
            config.layer_types = new_layer_types

        logger.info(f"Updated model config: {original_layers} -> {new_num_layers} layers")
        if hasattr(config, 'layer_types'):
            logger.debug(f"Updated layer_types length: {len(config.layer_types)}")

    def save_pruned_model(self, output_path: str):
        """Save the pruned model with proper layer remapping"""
        import tempfile
        import shutil
        from pathlib import Path
        
        # Create a temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # First save to temporary directory
            self.model.save_pretrained(temp_dir)
            self.tokenizer.save_pretrained(temp_dir)
            
            # Read and fix the model index file if it exists
            index_path = Path(temp_dir) / "model.safetensors.index.json"
            if index_path.exists():
                self._fix_model_index(str(index_path))
            
            # Copy to final output path
            Path(output_path).mkdir(parents=True, exist_ok=True)
            shutil.copytree(temp_dir, output_path, dirs_exist_ok=True)
    
    def _fix_model_index(self, index_path: str):
        """Fix layer indices in model.safetensors.index.json after pruning"""
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        # Create mapping from original layer indices to new indices
        old_to_new_mapping = {}
        for new_idx in range(len(self.layers)):
            if hasattr(self.layers[new_idx], '_original_index'):
                old_idx = self.layers[new_idx]._original_index
                old_to_new_mapping[old_idx] = new_idx
            else:
                # Fallback: assume sequential if no original index
                old_to_new_mapping[new_idx] = new_idx
        
        # Process weight mappings
        new_weight_map = {}
        
        for weight_name, file_name in index_data["weight_map"].items():
            # Check if this is a layer weight
            layer_match = re.match(r'model\.layers\.(\d+)\.(.+)', weight_name)
            if layer_match:
                old_layer_idx = int(layer_match.group(1))
                weight_suffix = layer_match.group(2)
                
                # Map to new layer index if layer still exists
                if old_layer_idx in old_to_new_mapping:
                    new_layer_idx = old_to_new_mapping[old_layer_idx]
                    new_weight_name = f"model.layers.{new_layer_idx}.{weight_suffix}"
                    new_weight_map[new_weight_name] = file_name
                # Otherwise skip this weight (layer was pruned)
            else:
                # Non-layer weights (embed_tokens, norm, lm_head) remain unchanged
                new_weight_map[weight_name] = file_name
        
        # Update the index data
        index_data["weight_map"] = new_weight_map
        
        # Write back the fixed index
        with open(index_path, 'w') as f:
            json.dump(index_data, f, indent=2)

    def compute_metric(self, hiddens: List[torch.Tensor], metric: str = "block_influence"):
        """
        Compute importance metric for hidden states

        Args:
            hiddens: List of hidden states from model layers
            metric: Metric name to compute
        """
        from .metrics import get_metric_function

        metric_func = get_metric_function(metric)
        n = 1

        for i in range(len(hiddens) - n):
            in_hidden = hiddens[i]
            out_hidden = hiddens[i+n]
            importance_score = metric_func(in_hidden, out_hidden)
            self.importances[i] += importance_score.mean().cpu().item()

    @torch.inference_mode()
    def eval_importance(
        self,
        prompts: List[str],
        metric: str = "block_influence"
    ):
        """Evaluate layer importance using specified metric"""
        # Reset importance scores
        self.importances = [0.0 for _ in self.layers]

        logger.info(f"Evaluating layer importance using {metric} metric on {len(prompts)} samples...")

        for i, prompt in enumerate(prompts):
            if i % 20 == 0:  # Progress logging every 20 samples
                logger.debug(f"  Processing sample {i+1}/{len(prompts)}...")

            # Tokenize individual sample
            inputs = self.tokenizer(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=4096,  # Handle up to 4k characters worth of tokens
                padding=False
            )

            try:
                # Forward pass to get hidden states
                outputs = self.model(
                    input_ids=inputs.input_ids.to(self.model.device),
                    attention_mask=inputs.attention_mask.to(self.model.device),
                    output_hidden_states=True,
                )

                # Compute metric for this sample
                self.compute_metric(outputs.hidden_states, metric)

            except Exception as e:
                logger.warning(f"  Failed to process sample {i+1} ({len(prompt)} chars): {e}")
                continue

        # Normalize importance scores by number of successfully processed samples
        total_samples = len(prompts)
        if total_samples > 0:
            self.importances = [score / total_samples for score in self.importances]

        logger.success(f"✅ Completed layer importance evaluation with {metric} metric")