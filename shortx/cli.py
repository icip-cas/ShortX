#!/usr/bin/env python3
"""
Main CLI interface for ShortX
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="ShortX - A unified pruning toolkit for various AI model architectures",
        epilog="Use 'shortx <command> --help' for more information on a specific command."
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # ShortGPT subcommand
    shortgpt_parser = subparsers.add_parser(
        "shortgpt",
        help="Use ShortGPT for LLM layer pruning"
    )
    shortgpt_parser.add_argument(
        "action",
        choices=["analyze", "generate", "demo"],
        help="Action to perform"
    )
    
    # ShortV subcommand
    shortv_parser = subparsers.add_parser(
        "shortv",
        help="Use ShortV for VLM optimization"
    )
    shortv_parser.add_argument(
        "action",
        choices=["chat", "analyze"],
        help="Action to perform"
    )
    
    # Info subcommand
    info_parser = subparsers.add_parser(
        "info",
        help="Show information about ShortX"
    )
    
    args = parser.parse_args()
    
    if args.command == "shortgpt":
        from .shortgpt.cli import main as shortgpt_main
        sys.argv = ["shortgpt", args.action] + sys.argv[3:]
        shortgpt_main()
    elif args.command == "shortv":
        from .shortv.cli import main as shortv_main
        sys.argv = ["shortv", args.action] + sys.argv[3:]
        shortv_main()
    elif args.command == "info":
        print("""
ShortX v0.1.0

A unified optimization toolkit for AI models. Currently includes ShortGPT and ShortV for efficient layer pruning, with more tools like ShortEmb coming soon.

Components:
- ShortGPT: Identifies and removes redundant layers in Large Language Models (LLMs)
- ShortV: Optimizes Vision-Language Models (VLMs) by selectively freezing visual token processing
- ShortEmb (coming soon): Embedding model optimization techniques
- More optimization tools in development...

For more information, visit: https://github.com/shortx/shortx
        """)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()