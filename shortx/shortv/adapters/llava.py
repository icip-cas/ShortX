"""
LLaVA model adapter for ShortV, providing concrete implementations for the ShortV interface.
"""
import os
import time
import torch
import types
import numpy as np
import pandas as pd
import torch.nn.functional as F
from io import BytesIO
from PIL import Image
from typing import List, Optional, Dict, Any, Callable, Union, Tuple
from transformers import AutoProcessor, LlavaForConditionalGeneration

from ..core import ShortV
from ...auto import ModelRegistry
from ...utils.logger import logger
from ...utils.patch import BasePatcher


class LlamaPatcher(BasePatcher):
    """Patcher for LLaMA-like models."""
    def _get_layers(self) -> torch.nn.ModuleList:
        """Accesses the decoder layers of a LLaMA model."""
        return self.model.model.layers

class LLaVAShortV(ShortV):
    """
    LLaVA-specific ShortV implementation.

    This adapter enables visual token skipping for LLaVA models by dynamically
    patching the model's forward methods at runtime.
    """
    def __init__(self, model_name: str, n_skip_layers: Optional[int] = None,
                 skip_layers: Optional[List[int]] = None, **kwargs):
        """
        Initialize LLaVA ShortV, using pre-computed configs if available.
        """
        super().__init__(model_name, n_skip_layers, skip_layers, **kwargs)
        self._original_llava_forward = None
        self._patching_enabled = False
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name,
            device_map=self.device_map,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self.patcher = LlamaPatcher(self.model.language_model)
        if n_skip_layers is not None and skip_layers is not None:
            assert n_skip_layers <= len(skip_layers), "n_skip_layers must be less than or equal to the length of skip_layers."
        elif n_skip_layers is not None and skip_layers is None:
            self.enable_monkey_patching()
            result_dic = self.calculate_lc_scores()
            self.set_skip_layers(result_dic['recommended_skip_layers'])

    def _create_patched_llava_forward(self):
        """Creates a patched forward method for LlavaForConditionalGeneration that preserves the original signature."""
        original_forward = self.model.forward

        # By defining the wrapper with the exact same signature as the original, we prevent `generate` from breaking.
        # This signature is updated to match the user's newer version of transformers.
        def patched_forward(
            self_model,
            input_ids: torch.LongTensor = None,
            pixel_values: torch.FloatTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            vision_feature_layer: Optional[Union[int, List[int]]] = None,
            vision_feature_select_strategy: Optional[str] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            image_sizes: torch.Tensor = None,
            **lm_kwargs,
        ):
            # Bundle all arguments into a kwargs dict to easily pass them down.
            kwargs = {
                "input_ids": input_ids,
                "pixel_values": pixel_values,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "inputs_embeds": inputs_embeds,
                "vision_feature_layer": vision_feature_layer,
                "vision_feature_select_strategy": vision_feature_select_strategy,
                "labels": labels,
                "use_cache": use_cache,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
                "cache_position": cache_position,
                "logits_to_keep": logits_to_keep,
                "image_sizes": image_sizes,
            }
            # Add any extra keyword arguments from **lm_kwargs
            kwargs.update(lm_kwargs)

            # --- Custom Logic: Start ---
            # Calculate and inject `image_token_indices` for our decoder layer patches.
            # This is cached to work correctly with `generate`.
            current_input_ids = input_ids if input_ids is not None else inputs_embeds
            if current_input_ids is not None and current_input_ids.shape[1] > 1:
                image_token_indices = torch.where(input_ids == self_model.config.image_token_index)[1]
                self_model._image_token_indices_cache = image_token_indices

            if hasattr(self_model, '_image_token_indices_cache'):
                kwargs['image_token_indices'] = self_model._image_token_indices_cache
            # --- Custom Logic: End ---

            # Call the original forward method. Our decoder layer patches will handle `image_token_indices`.
            return original_forward(**kwargs)

        return types.MethodType(patched_forward, self.model)

    def _create_patched_forward(self, layer_idx: int, original_forward: Callable):
        """Creates a patched forward method for a single decoder layer, preserving the original signature."""

        # By defining the wrapper with the exact same signature as the original decoder layer,
        # we ensure that `generate` can correctly pass all arguments, especially `cache_position`.
        def patched_decoder_layer_forward(
            self_layer,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Any] = None,  # Use Any for Cache to avoid import issues
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs, # Captures image_token_indices and other potential kwargs like position_embeddings
        ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

            image_token_indices = kwargs.get("image_token_indices")

            # CRITICAL FIX: The original forward method does not accept our custom 'image_token_indices'.
            # We must remove it before calling the original method.
            original_kwargs = {k: v for k, v in kwargs.items() if k != 'image_token_indices'}
            # --- Layer Skipping Logic ---
            if layer_idx in self.skip_layers and image_token_indices is not None and image_token_indices.numel() > 0:
                if hidden_states.shape[1] > 1:
                    output = original_forward(
                        hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                        past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache,
                        cache_position=cache_position, **original_kwargs
                    )
                    final_hidden_states = output[0]
                    final_hidden_states[:, image_token_indices, :] = hidden_states[:, image_token_indices, :]
                    return (final_hidden_states,) + output[1:]

            # For single-token generation or non-skipped layers, run the original forward pass.
            return original_forward(
                hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache,
                cache_position=cache_position, **original_kwargs
            )
        return patched_decoder_layer_forward

    def enable_monkey_patching(self):
        """Enables monkey patching for LLaVA models."""
        if self._patching_enabled:
            return

        # 1. Patch LlavaForConditionalGeneration.forward to pass image_token_indices
        if not self._original_llava_forward:
            self._original_llava_forward = self.model.forward
            self.model.forward = self._create_patched_llava_forward()
            logger.info("Patched LlavaForConditionalGeneration.forward to pass image token indices.")

        # 2. Patch specified decoder layers
        for layer_idx in range(len(self.patcher._get_layers())):
            original_forward = self.patcher._original_forwards.get(layer_idx, self.patcher._get_layers()[layer_idx].forward)
            new_forward = self._create_patched_forward(layer_idx, original_forward)
            self.patcher.patch_layer(layer_idx, new_forward, strategy_name=f"shortv_patched_layer_{layer_idx}")

        logger.info(f"Enabled ShortV patching for layers: {self.skip_layers or 'None'}")
        self._patching_enabled = True

    def disable_monkey_patching(self):
        """Disables monkey patching and invalidates the model to force a reload."""
        if not self._patching_enabled:
            return

        # Restore original Llava forward if it was patched
        if self._original_llava_forward:
            self.model.forward = self._original_llava_forward
            self._original_llava_forward = None

        # Unpatch all decoder layers
        if self.patcher:
            self.patcher.unpatch_all()

        self._patching_enabled = False
        logger.info("Disabled monkey patching. Model will be reloaded on next use.")

    def set_skip_layers(self, layers: List[int]):
        """
        Sets the layers to be replaced/skipped.
        This dynamically updates the skipping behavior if patching is already enabled.
        """
        self.skip_layers = layers
        logger.info(f"Updated ShortV skip layers to: {self.skip_layers or 'None'}")

    def case_forward(self, sample: Optional[pd.Series] = None):

        question = sample['question']
        options = sample['options']
        image_dict = sample['image_1']
        pil_image = Image.open(BytesIO(image_dict['bytes']))
        prompt_text = f"{question}\n{options}"

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)
        output = self.model(**inputs)
        return output

    def calculate_lc_scores(self, ) -> Dict[str, Any]:
        """
        Calculates Layer Contribution (LC) scores for LLaVA models using KL-Divergence.
        This method is computationally intensive as it iterates over the dataset and performs N+1 forward passes per sample.
        """
        original_skip_layers = self.skip_layers.copy()

        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "mmmlu_dev.parquet")
        df = pd.read_parquet(data_path, engine='pyarrow')[:10]
        num_samples = len(df)
        logger.info(f"Found {num_samples} samples for evaluation.")

        num_layers = len(self.patcher._get_layers())
        mean_lc_scores = np.zeros(num_layers)
        
        kl_div_loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)

        try:
            start_time = time.time()
            # Iterate over each sample in the dataset
            for i, sample in df.iterrows():
                # --- 1. Baseline run for the current sample (no layers skipped) ---
                self.skip_layers = []
                with torch.no_grad():
                    baseline_outputs = self.case_forward(sample=sample)
                baseline_logits = baseline_outputs.logits[:, -1, :]
                baseline_log_probs = F.log_softmax(baseline_logits, dim=-1)

                # --- 2. Iterative runs for the current sample (skipping one layer at a time) ---
                sample_lc_scores = []
                for layer_idx in range(num_layers):
                    self.skip_layers = [layer_idx]
                    
                    with torch.no_grad():
                        skipped_outputs = self.case_forward(sample=sample)
                    skipped_logits = skipped_outputs.logits[:, -1, :]
                    skipped_log_probs = F.log_softmax(skipped_logits, dim=-1)

                    kl_divergence = kl_div_loss(skipped_log_probs, baseline_log_probs).item()
                    sample_lc_scores.append(kl_divergence)
                
                logger.debug(f"LC scores for sample {i+1}: {sample_lc_scores}")

                # --- Update running average and layer importance ranking in real-time ---
                samples_processed = i + 1
                # Update mean_lc_scores using the running average formula
                mean_lc_scores += (np.array(sample_lc_scores) - mean_lc_scores) / samples_processed
                
                # Real-time calculation and sorting
                current_importance_ranking = np.argsort(mean_lc_scores).tolist()

                # --- Time estimation and progress update ---
                elapsed_time = time.time() - start_time
                avg_time_per_sample = elapsed_time / samples_processed
                remaining_samples = num_samples - samples_processed
                estimated_remaining_time_seconds = remaining_samples * avg_time_per_sample
                
                # Format seconds into a more readable format (M:S or H:M:S)
                if estimated_remaining_time_seconds > 3600:
                    remaining_time_str = time.strftime('%H:%M:%S', time.gmtime(estimated_remaining_time_seconds))
                else:
                    remaining_time_str = time.strftime('%M:%S', time.gmtime(estimated_remaining_time_seconds))

                logger.info(
                    f"Sample {samples_processed}/{num_samples} | "
                    f"ETA: {remaining_time_str} | "
                    f"Current layer importance ranking: {current_importance_ranking}"
                )

            # After the loop, mean_lc_scores is the average for the entire dataset
            total_elapsed_time = time.time() - start_time
            logger.info(f"Processed all {num_samples} samples. Total time: {total_elapsed_time / 60:.2f} minutes.")
            final_mean_lc_scores = mean_lc_scores.tolist()

        except Exception as e:
            logger.error(f"An error occurred during LC score evaluation: {e}")
            raise
        finally:
            # Restore original state
            self.skip_layers = original_skip_layers
            logger.info("LC score evaluation finished.")

        # A higher KL divergence means the layer's removal causes a larger shift, thus it's more important.
        # We sort by score ascending, so less important layers come first.
        self.layers_importance = np.argsort(mean_lc_scores).tolist()

        num_layers = len(self.layers_importance)
        recommended_n = self.n_skip_layers if self.n_skip_layers is not None else (num_layers * 2) // 3
        logger.info(f"Final layer importance ranking: {self.layers_importance}")
        logger.info(f"Recommended layers to skip: {self.layers_importance[:recommended_n]}")
        return {
            "model": self.model_name,
            "total_layers": num_layers,
            "lc_scores": mean_lc_scores.tolist(), # Returns the average scores
            "layer_ranking_by_importance": self.layers_importance, # Lower score = less important = better to skip
            "recommended_skip_layers": self.layers_importance[:recommended_n],
        }
    
ModelRegistry.register_shortv("llava", LLaVAShortV)