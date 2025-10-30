#!/usr/bin/env python3
"""
CLI interface for ShortV - Visual token skipping for VLMs
"""

import os
import json
import argparse
import time
from datetime import datetime

from ..auto import AutoShortV
from ..utils.logger import logger


def analyze_command(args):
    """Analyze VLM layer importance for visual token processing"""
    logger.info(f"🔍 Loading VLM model: {args.model}")
    
    # Initialize ShortV with automatic adapter selection
    shortv = AutoShortV.from_pretrained(
        model_name_or_path=args.model,
        torch_dtype="auto",
        device_map="auto"
    )
    
    # Run layer contribution analysis
    logger.info(f"📊 Starting layer contribution analysis...")
    logger.info(f"This will analyze layer importance by evaluating KL-divergence")
    logger.info(f"when each layer is skipped individually.")
    
    # Calculate LC scores
    shortv.enable_patching()
    results = shortv.calculate_lc_scores()
    
    # Print results
    logger.info("\n📈 LAYER CONTRIBUTION ANALYSIS RESULTS")
    logger.info(f"Model: {results['model']}")
    logger.info(f"Total layers: {results['total_layers']}")
    
    skip_layers = results['recommended_skip_layers']
    layer_ranking = results['layer_ranking_by_importance']
    lc_scores = results['lc_scores']
    
    logger.info(f"\n🎯 Layer importance ranking (from least to most important):")
    logger.info(f"   {layer_ranking}")
    
    logger.info(f"\n📊 Recommended layers to skip ({len(skip_layers)} layers):")
    logger.info(f"   {skip_layers}")
    
    logger.info("\n📊 Layer Contribution scores (lower = less important):")
    for i, score in enumerate(lc_scores):
        status = "🟡 SKIP" if i in skip_layers else "✅ KEEP"
        logger.info(f"   Layer {i:2d}: {score:8.6f} {status}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(args.model)
    filename = f"shortv_analysis_{model_name}_{timestamp}.json"
    filepath = os.path.join(args.output, filename)
    
    # Save results
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    logger.success(f"💾 Results saved to: {filepath}")


def chat_command(args):
    """Chat with VLM using optional layer skipping"""
    logger.info(f"🔍 Loading VLM model: {args.model}")
    
    # Parse skip layers if provided
    skip_layers = []
    if args.skip_layers:
        skip_layers = [int(x.strip()) for x in args.skip_layers.split(',')]
        logger.info(f"🎯 Using specified skip layers: {skip_layers}")
    
    # Initialize ShortV
    shortv = AutoShortV.from_pretrained(
        model_name_or_path=args.model,
        n_skip_layers=len(skip_layers) if skip_layers else None,
        torch_dtype="auto",
        device_map="auto"
    )
    
    # Enable patching and set skip layers if provided
    if skip_layers:
        shortv.enable_patching()
        shortv.set_skip_layers(skip_layers)
        logger.info(f"✅ Layer skipping enabled for {len(skip_layers)} layers")
    else:
        logger.info("✅ Model loaded without layer skipping")
    
    # Ensure we have an image for VLM chat
    if not args.image:
        logger.error("❌ Image is required for VLM chat. Use --image <path>")
        return
    
    if not os.path.exists(args.image):
        logger.error(f"❌ Image file not found: {args.image}")
        return
    
    # Prepare conversation
    logger.info(f"   💬 Chat Configuration:")
    logger.info(f"   Prompt: {args.prompt}")
    logger.info(f"   Image: {args.image}")
    if skip_layers:
        logger.info(f"   Skip layers: {skip_layers}")
    logger.info(f"   Max tokens: {args.max_tokens}")
    
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": args.image},
                {"type": "text", "text": args.prompt},
            ],
        },
    ]
    
    # Process input
    inputs = shortv.processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(shortv.model.device)
    
    # Generate response
    logger.info("🤖 Generating response...")
    
    try:
        generate_ids = shortv.model.generate(
            **inputs, 
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            do_sample=args.temperature > 0
        )
        
        # Decode response
        response = shortv.processor.decode(generate_ids[0, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        
        # Print results
        logger.info("📝 Response:")
        logger.success(f"{response}")
        
        
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()




def main():
    parser = argparse.ArgumentParser(description="ShortV - Visual token skipping for VLMs")
    
    subparsers = parser.add_subparsers(dest="action", help="Action to perform")
    
    # Analyze command - analyze visual token importance
    analyze_parser = subparsers.add_parser("analyze", help="Analyze layer importance and get ranking")
    analyze_parser.add_argument("--model", required=True, help="HuggingFace VLM model path or name")
    analyze_parser.add_argument("--output", default="shortv_analysis_results",
                               help="Output directory for analysis results")
    
    # Chat command - chat with VLM using optional layer skipping
    chat_parser = subparsers.add_parser("chat", help="Chat with VLM using optional layer skipping")
    chat_parser.add_argument("--model", required=True, help="HuggingFace VLM model path or name")
    chat_parser.add_argument("--prompt", required=True, help="Input prompt for generation")
    chat_parser.add_argument("--image", required=True, help="Path to image file")
    chat_parser.add_argument("--skip-layers", 
                            help="Comma-separated layer indices to skip (e.g., '0,2,4,6,8')")
    chat_parser.add_argument("--max-tokens", type=int, default=100,
                            help="Maximum tokens to generate (default: 100)")
    chat_parser.add_argument("--temperature", type=float, default=0.7,
                            help="Temperature for generation (default: 0.7, set to 0 for deterministic)")
    
    
    args = parser.parse_args()
    
    if args.action == "analyze":
        analyze_command(args)
    elif args.action == "chat":
        chat_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()