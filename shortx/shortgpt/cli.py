#!/usr/bin/env python3
"""
CLI interface for ShortGPT
"""
import os
import sys
import json
import torch
import argparse
from datetime import datetime

from shortx.auto import AutoShortGPT
from shortx.utils.patch import PruningStrategy
from shortx.utils.logger import logger


def analyze_command(args):
    """Analyze model layer importance"""
    logger.info(f"🔍 Loading model: {args.model}")
    
    # Load model using AutoShortGPT
    model = AutoShortGPT.from_pretrained(args.model)
    
    # Set number of layers to prune if specified
    if args.n_prune_layers:
        model.n_prune_layers = args.n_prune_layers
    
    # Prepare dataset samples
    dataset_samples = None
    if args.dataset:
        try:
            with open(args.dataset, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                dataset_samples = [json.loads(line.strip()).get('text', '') 
                                 for line in lines[:args.num_samples]]
                dataset_samples = [text for text in dataset_samples if text]
        except Exception as e:
            logger.warning(f"Could not load dataset from {args.dataset}: {e}")
            logger.info("Using default samples instead...")
    
    # Run analysis
    logger.info(f"📊 Starting layer importance analysis using {args.metric} metric...")
    results = model.analyze_layers(
        dataset_samples=dataset_samples,
        metric=args.metric
    )
    
    # Print results
    logger.info("📈 LAYER IMPORTANCE ANALYSIS RESULTS")
    logger.info(f"Model: {results['model']}")
    logger.info(f"Total layers: {results['total_layers']}")
    logger.info(f"Metric used: {results['metric_used']}")
    logger.info(f"Recommended layers to prune: {len(results['recommended_prune_layers'])}")
    
    logger.info("\n🎯 Recommended layers to prune:")
    prune_layers = results['recommended_prune_layers']
    logger.info(f"   {prune_layers}")
    
    logger.info("\n📊 Layer importance scores (lower = less important):")
    for i, score in enumerate(results['importances']):
        status = "🔴 PRUNE" if i in prune_layers else "✅ KEEP "
        logger.info(f"   Layer {i:2d}: {score:8.4f} {status}")
    

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = results['model'].replace('/', '_').replace('\\', '_')
    filename = f"analysis_{model_name}_{timestamp}.json"
    filepath = os.path.join(args.output, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    logger.success(f"💾 Results saved to: {filepath}")


def demo_prune_command(args):
    """Demo pruning effects by comparing original vs pruned generation"""
    logger.info(f"🔍 Loading model: {args.model}")
    
    # Load model using AutoShortGPT
    model = AutoShortGPT.from_pretrained(args.model)
    
    # Determine layers to prune
    layers_to_prune = None
    if args.layers_to_prune:
        layers_to_prune = [int(x) for x in args.layers_to_prune.split(',')]
        logger.info(f"🎯 Using manually specified layers: {layers_to_prune}")
    else:
        logger.info(f"📊 Running analysis to determine layers to prune...")
        if args.n_prune_layers:
            model.n_prune_layers = args.n_prune_layers
        
        analysis = model.analyze_layers(metric=args.metric)
        layers_to_prune = analysis['recommended_prune_layers']
        logger.info(f"🎯 Analysis recommends pruning layers: {layers_to_prune}")
    
    # Generate original text
    logger.info("🔄 GENERATING WITH ORIGINAL MODEL")
    logger.info(f"Prompt: {args.prompt}")
    
    inputs = model.tokenizer(args.prompt, return_tensors='pt').to(model.model.device)
    
    try:
        with torch.no_grad():
            original_output = model.model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                temperature=0.7,
                do_sample=True,
            )
        
        original_text = model.tokenizer.decode(original_output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        logger.success(f"Original: {original_text}")
        
    except Exception as e:
        logger.error(f"Error generating with original model: {e}")
        return
    
    # Enable monkey patching and prune layers
    logger.info("🔄 GENERATING WITH PRUNED MODEL (MONKEY PATCHING)")
    
    try:
        model.prune_layers_patch(layers_to_prune, PruningStrategy.IDENTITY)
        
        logger.info(f"🎯 Pruned {len(layers_to_prune)} layers using monkey patching")
        logger.info(f"Prompt: {args.prompt}")
            
        # Generate with pruned model
        with torch.no_grad():
            pruned_output = model.model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=model.tokenizer.eos_token_id
            )
        
        pruned_text = model.tokenizer.decode(pruned_output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        logger.success(f"Pruned:   {pruned_text}")
        logger.info(f"\n📊 COMPARISON SUMMARY")
        logger.info(f"Layers pruned: {len(layers_to_prune)} / {len(model.layers)}")
        logger.info(f"Model reduction: ~{len(layers_to_prune)/len(model.layers)*100:.1f}%")
        
    except Exception as e:
        logger.error(f"Error during pruned generation: {e}")
        
    finally:
        # Always restore model to original state
        if hasattr(model, 'disable_monkey_patching'):
            model.disable_monkey_patching()
            logger.success("✅ Model restored to original state")


def save_pruned_command(args):
    """Save physically pruned model"""
    logger.info(f"🔍 Loading model: {args.model}")
    
    # Load model using AutoShortGPT  
    model = AutoShortGPT.from_pretrained(args.model)
    
    # Determine layers to prune
    layers_to_prune = None
    if args.layers_to_prune:
        layers_to_prune = [int(x) for x in args.layers_to_prune.split(',')]
        logger.info(f"🎯 Using manually specified layers: {layers_to_prune}")
    else:
        logger.info(f"📊 Running analysis to determine layers to prune...")
        if args.n_prune_layers:
            model.n_prune_layers = args.n_prune_layers
        
        analysis = model.analyze_layers(metric=args.metric)
        layers_to_prune = analysis['recommended_prune_layers']
        logger.info(f"🎯 Analysis recommends pruning layers: {layers_to_prune}")
    
    # Perform physical layer removal
    logger.info(f"🔪 Performing physical layer removal...")
    logger.info(f"Removing {len(layers_to_prune)} layers: {layers_to_prune}")
    
    try:
        removed_layers = model.prune_layers(layers_to_prune)
        logger.success(f"✅ Successfully removed layers: {removed_layers}")
        
        # Save pruned model
        # Create output directory if it doesn't exist
        os.makedirs(args.output_path, exist_ok=True)
        
        # Generate model directory name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = args.model.split('/')[-1].replace('/', '_').replace('\\', '_')
        model_dir = f"pruned_{model_name}_{len(layers_to_prune)}layers_{timestamp}"
        full_output_path = os.path.join(args.output_path, model_dir)
        
        logger.info(f"\n💾 Saving pruned model to: {full_output_path}")
        model.save_pruned_model(full_output_path)
        
        logger.info("\n📊 PRUNING SUMMARY")
        logger.info(f"Original layers: {len(model.layers) + len(removed_layers)}")
        logger.info(f"Pruned layers: {len(model.layers)}")
        logger.info(f"Removed layers: {len(removed_layers)}")
        logger.info(f"Model reduction: ~{len(removed_layers)/(len(model.layers) + len(removed_layers))*100:.1f}%")
        logger.success(f"Saved to: {full_output_path}")
        
    except Exception as e:
        logger.error(f"Error during pruning or saving: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="ShortGPT - Efficient layer pruning for LLMs")
    
    subparsers = parser.add_subparsers(dest="action", help="Action to perform")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze layer importance")
    analyze_parser.add_argument("--model", required=True, help="HuggingFace model name")
    analyze_parser.add_argument("--metric", default="block_influence", 
                               help="Metric to use for importance analysis")
    analyze_parser.add_argument("--n-prune-layers", type=int, 
                               help="Number of layers to recommend for pruning")
    analyze_parser.add_argument("--dataset", 
                               help="Path to dataset file (JSONL format)")
    analyze_parser.add_argument("--num-samples", type=int, default=100,
                               help="Number of samples to use for analysis")
    analyze_parser.add_argument("--output", default="shortgpt_analysis_results",
                               help="Output directory to save analysis results")
    
    # Demo-prune command  
    demo_parser = subparsers.add_parser("demo-prune", help="Demo pruning effects")
    demo_parser.add_argument("--model", required=True, help="HuggingFace model name")
    demo_parser.add_argument("--prompt", required=True, help="Input prompt for generation")
    demo_parser.add_argument("--layers-to-prune", 
                            help="Comma-separated layer indices (e.g., '5,10,15')")
    demo_parser.add_argument("--metric", default="block_influence",
                            help="Metric for automatic layer selection")
    demo_parser.add_argument("--n-prune-layers", type=int,
                            help="Number of layers to prune (if not specifying manually)")
    demo_parser.add_argument("--max-tokens", type=int, default=100,
                            help="Maximum tokens to generate")
    
    # Save-pruned command
    save_parser = subparsers.add_parser("save-pruned", help="Save physically pruned model")
    save_parser.add_argument("--model", required=True, help="HuggingFace model name")
    save_parser.add_argument("--output-path", default="shortgpt_pruned_models", help="Directory path to save pruned model")
    save_parser.add_argument("--layers-to-prune",
                            help="Comma-separated layer indices (e.g., '5,10,15')")
    save_parser.add_argument("--metric", default="block_influence",
                            help="Metric for automatic layer selection")
    save_parser.add_argument("--n-prune-layers", type=int,
                            help="Number of layers to prune (if not specifying manually)")
    
    args = parser.parse_args()
    
    if args.action == "analyze":
        analyze_command(args)
    elif args.action == "demo-prune":
        demo_prune_command(args)
    elif args.action == "save-pruned":
        save_pruned_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()