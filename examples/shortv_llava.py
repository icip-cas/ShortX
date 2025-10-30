import os
from shortx import AutoShortV
# --- Configuration ---
# Model path and image path, please modify according to your environment
MODEL_PATH = "llava-hf/llava-1.5-7b-hf"
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "cartoon_image.jpg")

# Number of layers to skip. LLaVAShortV will use its pre-computed optimal layer list.
N_SKIP_LAYERS = 10

# 1. Initialize ShortV
print("Initializing ShortV...")
shortv = AutoShortV.from_pretrained(
    model_name_or_path=MODEL_PATH,
    n_skip_layers=N_SKIP_LAYERS,
    torch_dtype="auto",
    device_map="cuda:0"
)

# 2. Enable ShortV patching
# This will load the model and apply layer skipping optimization.
print(f"\nEnabling ShortV patching, skipping {N_SKIP_LAYERS} layers...")
shortv.enable_monkey_patching()

# 3. Prepare input
# Use shortv.processor to process input.
# Note: `apply_chat_template` will automatically handle image paths.
print("\nPreparing input...")
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": IMAGE_PATH},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]

# processor and model have already been loaded by shortv
inputs = shortv.processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(shortv.model.device)

# 4. Generate using optimized model
print("\nGenerating text using ShortV-enabled model...")
generate_ids_with_shortv = shortv.model.generate(**inputs, max_new_tokens=100)
decoded_with_shortv = shortv.processor.batch_decode(generate_ids_with_shortv, skip_special_tokens=True)
print("--- Output with ShortV ---")
print(decoded_with_shortv[0].strip())
print("-" * 26)


# 5. Dynamically change number of layers to skip
new_skip_layers = 20
print(f"\nDynamically changing number of layers to skip to {new_skip_layers}...")
# Get the first new_skip_layers layers from pre-computed configuration
shortv.set_skip_layers(shortv.layers_importance[:new_skip_layers])

print(f"\nGenerating text again with {new_skip_layers} skipped layers...")
generate_ids_new = shortv.model.generate(**inputs, max_new_tokens=100)
decoded_new = shortv.processor.batch_decode(generate_ids_new, skip_special_tokens=True)
print(f"--- Output with {new_skip_layers} skipped layers ---")
print(decoded_new.strip())
print("-" * 26)


# 6. Restore original model for comparison
print("\nTesting original model performance by setting skip layer list to empty...")
shortv.disable_monkey_patching() # Dynamically set to not skip any layers
generate_ids_original = shortv.model.generate(**inputs, max_new_tokens=100)
print("\nGenerating text using original model (no layer skipping)...")
decoded_original = shortv.processor.batch_decode(generate_ids_original, skip_special_tokens=True)
print("--- Original model output ---")
print(decoded_original.strip())
print("-" * 32)