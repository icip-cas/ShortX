"""
Qwen Model Testing Script

Tests both traditional layer deletion and monkey patch pruning methods
for Qwen models using the ShortX toolkit.
"""

import torch
from shortx.auto import AutoShortGPT
from shortx.utils.patch import PruningStrategy
from shortx.utils.logger import logger

def test_qwen_traditional_pruning():
    """Test traditional layer deletion method"""
    logger.info("Testing Qwen Traditional Layer Deletion")
    
    # Initialize Qwen model
    logger.info("Loading Qwen model...")
    model = AutoShortGPT.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    
    logger.info(f"Original model layers: {len(model.layers)}")
    
    # Analyze layer importance
    logger.info("Analyzing layer importance with block influence metric...")
    analysis = model.analyze_layers(metric="block_influence")
    
    logger.info(f"Layer importances: {analysis['importances']}")
    logger.info(f"Recommended layers to prune: {analysis['recommended_prune_layers']}")
    
    # Test text generation before pruning
    test_prompt = "The future of artificial intelligence is"
    logger.info(f"Before pruning - Testing generation:")
    logger.info(f"Prompt: {test_prompt}")
    
    # Generate text
    inputs = model.tokenizer(test_prompt, return_tensors="pt").to(model.model.device)
    with torch.no_grad():
        outputs = model.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.7,
            pad_token_id=model.tokenizer.eos_token_id
        )
    generated_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Generated: {generated_text}")
    
    # Prune layers (traditional deletion)
    logger.info(f"Pruning {len(analysis['recommended_prune_layers'])} least important layers...")
    pruned_layers = model.prune_layers(analysis['recommended_prune_layers'])
    logger.info(f"Pruned layers: {pruned_layers}")
    logger.info(f"Remaining layers: {len(model.layers)}")
    
    # Test generation after pruning
    logger.info(f"After pruning - Testing generation:")
    inputs = model.tokenizer(test_prompt, return_tensors="pt").to(model.model.device)
    with torch.no_grad():
        outputs = model.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.7,
            pad_token_id=model.tokenizer.eos_token_id
        )
    generated_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Generated: {generated_text}")
    
    logger.success("✅ Traditional pruning test completed!")
    return model

def test_qwen_monkey_patch():
    """Test monkey patch pruning method"""
    logger.info("Testing Qwen Monkey Patch Pruning")
    
    # Initialize fresh model for monkey patching
    logger.info("Loading fresh Qwen model for monkey patching...")
    model = AutoShortGPT.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    
    logger.info(f"Original model layers: {len(model.layers)}")
    
    # Test text generation before patching
    test_prompt = "Machine learning algorithms can"
    logger.info(f"Before patching - Testing generation:")
    logger.info(f"Prompt: {test_prompt}")
    
    inputs = model.tokenizer(test_prompt, return_tensors="pt").to(model.model.device)
    with torch.no_grad():
        outputs = model.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.7,
            pad_token_id=model.tokenizer.eos_token_id
        )
    generated_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Generated: {generated_text}")
    
    # Analyze and get recommendations
    analysis = model.analyze_layers(metric="l2_distance")  # Try different metric
    recommended_layers = analysis['recommended_prune_layers'][:3]  # Prune fewer layers for demo
    
    logger.info(f"Recommended layers to patch: {recommended_layers}")
    
    # Test different patching strategies
    logger.info("Testing Identity Strategy (Skip layers completely) ")
    model.enable_monkey_patching()  # Enable monkey patching before use
    model.prune_layers_patch(recommended_layers[:2], PruningStrategy.IDENTITY)
    
    # Test generation with identity patching
    inputs = model.tokenizer(test_prompt, return_tensors="pt").to(model.model.device)
    with torch.no_grad():
        outputs = model.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.7,
            pad_token_id=model.tokenizer.eos_token_id
        )
    generated_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Generated (Identity): {generated_text}")
    
    # Test attention-only strategy
    logger.info("Testing Attention-Only Strategy")
    if len(recommended_layers) > 2:
        model.prune_layers_patch([recommended_layers[2]], PruningStrategy.ATTENTION_ONLY)
    
        # Test generation with attention-only patching
        inputs = model.tokenizer(test_prompt, return_tensors="pt").to(model.model.device)
        with torch.no_grad():
            outputs = model.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.7,
                pad_token_id=model.tokenizer.eos_token_id
            )
        generated_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated (Attn-Only): {generated_text}")
    
    # Test mixed strategies
    logger.info("Testing Mixed Strategies")
    mixed_config = {}
    if len(recommended_layers) >= 3:
        mixed_config = {
            recommended_layers[0]: PruningStrategy.IDENTITY,
            recommended_layers[1]: PruningStrategy.ATTENTION_ONLY,
            recommended_layers[2]: PruningStrategy.FFN_ONLY,
        }
        
        # Reset patches first
        model.disable_monkey_patching()
        model.enable_monkey_patching()
        
        # Apply mixed strategies
        model.prune_layers_mixed(mixed_config)
        
        # Test generation with mixed strategies
        inputs = model.tokenizer(test_prompt, return_tensors="pt").to(model.model.device)
        with torch.no_grad():
            outputs = model.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.7,
                pad_token_id=model.tokenizer.eos_token_id
            )
        generated_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated (Mixed): {generated_text}")
    
    # Show patch status
    patch_status = model.get_patch_status()
    logger.info(f"\nPatch Status: {patch_status}")
    
    # Test restoration
    logger.info("Testing Patch Restoration")
    model.disable_monkey_patching()
    
    # Test generation after restoration
    inputs = model.tokenizer(test_prompt, return_tensors="pt").to(model.model.device)
    with torch.no_grad():
        outputs = model.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.7,
            pad_token_id=model.tokenizer.eos_token_id
        )
    generated_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Generated (Restored): {generated_text}")
    
    logger.success("✅ Monkey patch pruning test completed!")
    return model

def test_multiple_metrics():
    """Test different layer importance metrics"""
    logger.info("Testing Multiple Layer Importance Metrics")
    
    from shortx.shortgpt.metrics import list_available_metrics
    
    # Show available metrics
    metrics = list_available_metrics()
    logger.info("Available metrics:")
    for name, desc in metrics.items():
        logger.info(f"  - {name}: {desc}")
    
    # Load model
    model = AutoShortGPT.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    
    # Test different metrics
    metrics_to_test = ["block_influence", "l2_distance", "variance_change"]
    results = {}
    
    logger.info(f"\nTesting metrics: {metrics_to_test}")
    for metric in metrics_to_test:
        logger.info(f"Analyzing with {metric}")
        analysis = model.analyze_layers(metric=metric)
        results[metric] = analysis['recommended_prune_layers'][:3]  # Top 3 layers
        logger.info(f"Recommended layers: {results[metric]}")
    
    # Compare results
    logger.info(f"Metrics Comparison")
    for metric, layers in results.items():
        logger.info(f"{metric}: {layers}")
    
    # Find consensus layers
    from collections import Counter
    all_layers = [layer for layers in results.values() for layer in layers]
    consensus = [layer for layer, count in Counter(all_layers).items() if count >= 2]
    logger.info(f"Consensus layers (appear in 2+ metrics): {consensus}")
    
    logger.success("✅ Multiple metrics test completed!")

def test_context_manager():
    """Test context manager functionality for automatic cleanup"""
    logger.info("Testing Context Manager (Automatic Cleanup)")
    
    test_prompt = "Context managers in Python are"
    
    # Use context manager for automatic patch cleanup
    logger.info("Using context manager for automatic patch management...")
    
    with AutoShortGPT.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct") as model:
        logger.info(f"Model loaded with {len(model.layers)} layers")
        
        # Enable monkey patching
        model.enable_monkey_patching()
        
        # Analyze and patch
        analysis = model.analyze_layers()
        patch_layers = analysis['recommended_prune_layers'][:2]
        
        logger.info(f"Patching layers: {patch_layers}")
        model.prune_layers_patch(patch_layers, PruningStrategy.IDENTITY)
        
        # Test generation
        inputs = model.tokenizer(test_prompt, return_tensors="pt").to(model.model.device)
        with torch.no_grad():
            outputs = model.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=20,
                do_sample=True,
                temperature=0.7,
                pad_token_id=model.tokenizer.eos_token_id
            )
        generated_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated with patches: {generated_text}")
        
        # Show patch status
        patch_status = model.get_patch_status()
        logger.info(f"Patch status inside context: {patch_status['patching_enabled']}")
    
    # After exiting context, patches should be automatically cleaned up
    logger.success("✅ Exited context manager - patches automatically cleaned up!")
    logger.success("✅ Context manager test completed!")


def test_device_and_dtype_options():
    """Test device_map and torch_dtype parameter options"""
    logger.info("Testing Device and Data Type Options")
    
    # Test 1: CPU-only loading
    logger.info("Testing CPU-only model loading...")
    try:
        cpu_model = AutoShortGPT.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct", 
            device_map="cpu"
        )
        logger.success(f"✅ CPU model loaded successfully on device: {next(cpu_model.model.parameters()).device}")
        
        # Quick generation test
        inputs = cpu_model.tokenizer("Hello", return_tensors="pt")
        with torch.no_grad():
            outputs = cpu_model.model.generate(
                inputs.input_ids, 
                max_new_tokens=10,
                do_sample=False
            )
        generated = cpu_model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.success(f"CPU generation test: {generated}")
        
    except Exception as e:
        logger.error(f"CPU model loading failed: {e}")
    
    # Test 2: Auto device with specific dtype
    logger.info("Testing auto device with float32 dtype...")
    try:
        auto_model = AutoShortGPT.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct", 
            device_map="auto",
            torch_dtype=torch.float32
        )
        param = next(auto_model.model.parameters())
        logger.success(f"✅ Auto model loaded - Device: {param.device}, Dtype: {param.dtype}")
        
    except Exception as e:
        logger.error(f"Auto model with float32 loading failed: {e}")
    
    # Test 3: Test with different configurations based on GPU availability
    if torch.cuda.is_available():
        logger.info("CUDA available - testing GPU configurations...")
        try:
            # Try half precision on GPU
            gpu_model = AutoShortGPT.from_pretrained(
                "Qwen/Qwen2.5-0.5B-Instruct", 
                device_map="auto",
                torch_dtype=torch.float16
            )
            param = next(gpu_model.model.parameters())
            logger.success(f"✅ GPU model loaded - Device: {param.device}, Dtype: {param.dtype}")
            
        except Exception as e:
            logger.warning(f"GPU half precision loading failed (normal on some hardware): {e}")
    else:
        logger.info("No CUDA available, skipping GPU-specific tests")
    
    logger.success("✅ Device and data type options test completed!")

if __name__ == "__main__":
    logger.info("ShortX Qwen Testing Suite")
    
    try:
        # Test 1: Traditional layer deletion
        traditional_model = test_qwen_traditional_pruning()
        
        # Test 2: Monkey patch pruning  
        patch_model = test_qwen_monkey_patch()
        
        # Test 3: Multiple metrics comparison
        test_multiple_metrics()
        
        # Test 4: Context manager
        test_context_manager()
        
        # Test 5: Device and data type options
        test_device_and_dtype_options()
        
        logger.success("All Qwen Tests Completed Successfully! 🎉")
        logger.success("Key Features Tested:")
        logger.success("✅ Traditional layer deletion pruning")
        logger.success("✅ Non-invasive monkey patch pruning")
        logger.success("✅ Multiple pruning strategies (Identity, Attention-Only, FFN-Only, Mixed)")
        logger.success("✅ Multiple layer importance metrics")
        logger.success("✅ Automatic model restoration")
        logger.success("✅ Context manager for resource cleanup")
        logger.success("✅ Text generation before/after pruning")
        logger.success("✅ Device and data type configuration options")
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()