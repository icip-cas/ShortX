<p align="center">
  <img src="logo/shortx.png" alt="ShortX Logo" style="width: 600px; height: 300px; object-fit: contain;"/>
</p>

# ShortX

A unified pruning toolkit for AI models. Currently includes **ShortGPT** and **ShortV** for efficient layer pruning.

## Overview

ShortX is an extensible toolkit that provides various AI model optimization techniques:

- **ShortGPT**: Identifies and removes redundant layers in Large Language Models (LLMs) to reduce model size and inference time while maintaining performance
- **ShortV**: Optimizes Vision-Language Models (VLMs) by selectively freezing visual token processing in ineffective layers
- More optimization tools in development...

## Installation

### From Source

Since ShortX is still under development, install from source:

```bash
# Clone the repository
git clone https://github.com/icip-cas/ShortX
cd shortx

# Install in development mode (recommended for development)
pip install -e .

# Or install directly
pip install .
```

## Usage

### ShortGPT

#### CLI Usage

```bash
# Analyze layer redundancy in an LLM
CUDA_VISIBLE_DEVICES=0 shortgpt analyze --model Qwen/Qwen2.5-1.5B-Instruct --n-prune-layers 5

# Generate text with a pruned model
CUDA_VISIBLE_DEVICES=0 shortgpt demo-prune --model Qwen/Qwen2.5-1.5B-Instruct \
    --prompt "The color of sky is" \
    --n-prune-layers 5

# Save pruned model to disk
CUDA_VISIBLE_DEVICES=0 shortgpt save-pruned --model Qwen/Qwen2.5-1.5B-Instruct \
    --n-prune-layers 5 \
    --output-path ./pruned_model
```

#### Python API

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from shortx import AutoShortGPT, PruningStrategy

# Initialize model with automatic adapter selection
model = AutoShortGPT.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
# Automatically selects QwenShortGPT adapter for Qwen models
# or LlamaShortGPT adapter for Llama models

# Analyze layer importance
analysis = model.analyze_layers()
print(f"Layer importance scores: {analysis['importances']}")
print(f"Recommended layers to prune: {analysis['recommended_prune_layers']}")

# Apply pruning with non-invasive patching
with model.enable_monkey_patching() as m:
    # Skip layers completely using IDENTITY strategy
    m.prune_layers_patch([27, 26, 25, 28, 24, 29, 23, 21, 22], PruningStrategy.IDENTITY)
    model_inputs = m.tokenizer(["Hello world"], return_tensors="pt").to(m.model.device)
    # Generate with pruned model
    result_ids = m.model.generate(**model_inputs, max_length=100)
    result = m.tokenizer.decode(result_ids[0, model_inputs.input_ids.shape[-1]:])
    print(result)
# Model automatically restored to original state

# Mixed pruning strategies for fine-grained control
layer_config = {
    10: PruningStrategy.IDENTITY,       # Skip completely
    15: PruningStrategy.ATTENTION_ONLY, # Keep only attention
    20: PruningStrategy.FFN_ONLY        # Keep only FFN
}
model.prune_layers_mixed(layer_config)
result = model.generate("The meaning of life is", max_length=50)
```

### ShortV

#### CLI Usage

```bash
# Analyze layer importance using KL-divergence
CUDA_VISIBLE_DEVICES=0 shortv analyze --model llava-hf/llava-1.5-7b-hf

# Chat with VLM without layer skipping
CUDA_VISIBLE_DEVICES=0 shortv chat --model llava-hf/llava-1.5-7b-hf \
    --prompt "What is shown in this image?" \
    --image examples/cartoon_image.jpg

# Chat with specific layers skipped (based on analysis results)
CUDA_VISIBLE_DEVICES=0 shortv chat --model llava-hf/llava-1.5-7b-hf \
    --prompt "Describe this image in detail" \
    --image examples/cartoon_image.jpg \
    --skip-layers 0,2,4,6,8,10,12,14,16,18
```

#### Python API

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from shortx import AutoShortV

# Initialize ShortV with automatic adapter selection
shortv = AutoShortV.from_pretrained(
    model_name="llava-hf/llava-1.5-7b-hf",
    torch_dtype="auto",
    device_map="cuda:0"
)

# Enable layer skipping optimization
shortv.enable_monkey_patching()

# Prepare conversation with image
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "examples/cartoon_image.jpg"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]

# Process inputs
inputs = shortv.processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(shortv.model.device)

# Generate
generate_ids = shortv.model.generate(**inputs, max_new_tokens=100)
decoded = shortv.processor.decode(generate_ids[0, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
print(f"Response: {decoded.strip()}")

# Analyze layer contribution for optimization
results = shortv.calculate_lc_scores()
print(f"Layer importance ranking: {results['layer_ranking_by_importance']}")
print(f"Recommended layers to skip: {results['recommended_skip_layers']}")

# Dynamically change skip layers
shortv.set_skip_layers([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

# Generate
generate_ids = shortv.model.generate(**inputs, max_new_tokens=100)
decoded = shortv.processor.decode(generate_ids[0, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
print(f"Response: {decoded.strip()}")

# Disable layer skipping to restore original model
shortv.disable_monkey_patching()
```

## Features

### Core Features
- **AutoModel-style adapter selection**: Automatically selects the best adapter based on model type
- **Non-invasive monkey patching**: Models can be pruned and restored without permanent changes  
- **Context manager support**: Safe patching with automatic restoration
- **Extensible architecture**: Easy to add new model architectures

### ShortGPT
- Layer importance analysis using Block Influence metric
- Support for multiple models with specialized adapters
- Multiple pruning strategies: Identity (skip), Attention-only, FFN-only
- Easy integration with HuggingFace models

### ShortV
- Visual token optimization for VLMs through layer skipping
- Support for LLaVA family models with enhanced adapter
- Layer Contribution (LC) score calculation using KL-divergence analysis
- Dynamic layer skipping: Enable/disable specific layers on-the-fly
- Flexible optimization: User-defined or automatically recommended skip layers

## Benchmarks

### ShortGPT Performance

Performance comparison on MMLU (multiple-choice tasks) and GSM8K (generative tasks) benchmarks with different pruning strategies:

| Model | Pruning Ratio | Strategy | MMLU | GSM8K |
|-------|---------------|----------|------|-------|
| Llama-2-7B | 0% | - | 45.9 | 17.74 |
| Llama-2-7B | 28% | IDENTITY | 44.5 | 1.90 |
| Llama-2-7B | 28% | LAST_ATTN_FFN_ONLY | **46.1** | **16.60** |
| Llama-2-13B | 0% | - | 55.7 | 32.07 |
| Llama-2-13B | 25% | IDENTITY | 54.1 | 1.67 |
| Llama-2-13B | 25% | LAST_ATTN_FFN_ONLY | **55.2** | **27.60** |


## Supported Models

### ShortGPT
- **Qwen series** - with QwenShortGPT adapter
- **Llama series**  - with LlamaShortGPT adapter

### ShortV
- **LLaVA-1.5 series** - with LLaVAShortV adapter
- **LLaVA-1.6 series (LLaVA-NeXT)** - with LLaVANeXTShortV adapter

## Adding New Models

To add support for a new model architecture (e.g., Mistral):

1. **Create adapter file**: `shortx/shortgpt/adapters/mistral.py`
2. **Implement patcher**:
   ```python
    def _mistral_attention_only_forward(self_layer, hidden_states, ...):
           # Model-specific attention-only implementation
   class MistralPatcher(BasePatcher):
       def _get_layers(self):
           return self.model.model.layers
   ```
3. **Implement adapter**:
   ```python
   class MistralShortGPT(ShortGPT):
       def __init__(self, model_name, **kwargs):
           super().__init__(model_name, layers_path="model.layers", **kwargs)
           self.patcher = None
   ```
4. **Register adapter**:
   ```python
   ModelRegistry.register_shortgpt("mistral", MistralShortGPT)
   ```
5. **Import in `__init__.py`**: Add to adapters module

That's it! The new model will work automatically with `AutoShortGPT.from_pretrained()`.

## Citation

If you use ShortX in your research, please cite:

```bibtex
@inproceedings{men-etal-2025-shortgpt,
    title = "{S}hort{GPT}: Layers in Large Language Models are More Redundant Than You Expect",
    author = "Men, Xin  and
      Xu, Mingyu  and
      Zhang, Qingyu  and
      Yuan, Qianhao  and
      Wang, Bingning  and
      Lin, Hongyu  and
      Lu, Yaojie  and
      Han, Xianpei  and
      Chen, Weipeng",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.1035/",
    doi = "10.18653/v1/2025.findings-acl.1035",
    pages = "20192--20204",
    ISBN = "979-8-89176-256-5",
}

@inproceedings{yuan2025shortv,
  title={ShortV: Efficient Multimodal Large Language Models by Freezing Visual Tokens in Ineffective Layers},
  author={Yuan, Qianhao and Zhang, Qingyu and Liu, Yanjiang and Chen, Jiawei and Lu, Yaojie and Lin, Hongyu and Zheng, Jia and Han, Xianpei and Sun, Le},
  booktitle={2025 IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```
