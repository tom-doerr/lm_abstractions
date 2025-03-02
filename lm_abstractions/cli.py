"""
Command Line Interface for LM Abstractions
----------------------------------------

Provides command-line tools for using the LM Abstractions package.
"""

import argparse
import sys
from typing import List, Dict, Any, Optional

from lm_abstractions.scoring import TextAttributeScorer
from lm_abstractions.generation import ContentGenerator
from lm_abstractions.optimization import ContentOptimizer


def score_text(args: argparse.Namespace) -> None:
    """
    Score text based on attributes.
    
    Args:
        args: Command-line arguments
    """
    # Parse positive and negative attributes
    positive_attrs = args.positive_attrs.split(',') if args.positive_attrs else ["concise", "clear", "informative"]
    negative_attrs = args.negative_attrs.split(',') if args.negative_attrs else ["verbose", "unclear", "misleading"]
    
    # Initialize the scorer
    scorer = TextAttributeScorer(
        model=args.model,
        positive_attrs=positive_attrs,
        negative_attrs=negative_attrs,
        temperature=args.temperature
    )
    
    # Score the text
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = args.text
    
    print(f"Scoring text: '{text[:100]}...' (truncated)" if len(text) > 100 else f"Scoring text: '{text}'")
    
    score, details = scorer.score_text(text)
    
    # Print the results
    print(f"\nOverall Score: {score:.4f}")
    
    print("\nPositive Attributes:")
    for attr, attr_score in details['positive_attributes'].items():
        print(f"  {attr}: {attr_score.score:.4f}")
    
    print("\nNegative Attributes:")
    for attr, attr_score in details['negative_attributes'].items():
        print(f"  {attr}: {attr_score.score:.4f}")
    
    print(f"\nPositive Mean: {details['positive_mean']:.4f}")
    print(f"Negative Mean: {details['negative_mean']:.4f}")


def generate_content(args: argparse.Namespace) -> None:
    """
    Generate content using a language model.
    
    Args:
        args: Command-line arguments
    """
    # Initialize the generator
    generator = ContentGenerator(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    # Generate content
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            prompt = f.read()
    else:
        prompt = args.prompt
    
    print(f"Generating content for prompt: '{prompt[:100]}...' (truncated)" if len(prompt) > 100 else f"Generating content for prompt: '{prompt}'")
    
    result = generator.generate(prompt)
    
    # Print the results
    print("\nGenerated content:")
    
    # Try to extract content from various possible field names
    content = None
    for field in ['content', 'response', 'output', 'text']:
        if hasattr(result, field):
            content = getattr(result, field)
            break
            
    if content:
        print(content)
    else:
        print(result)


def optimize_content(args: argparse.Namespace) -> None:
    """
    Optimize content generation based on scoring metrics.
    
    Args:
        args: Command-line arguments
    """
    # Parse positive and negative attributes
    positive_attrs = args.positive_attrs.split(',') if args.positive_attrs else ["concise", "clear", "informative"]
    negative_attrs = args.negative_attrs.split(',') if args.negative_attrs else ["verbose", "unclear", "misleading"]
    
    # Initialize the optimizer
    optimizer = ContentOptimizer(
        model=args.model,
        positive_attrs=positive_attrs,
        negative_attrs=negative_attrs,
        optimization_method=args.method,
        num_candidates=args.candidates
    )
    
    # Get the prompt
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            prompt = f.read()
    else:
        prompt = args.prompt
    
    print(f"Optimizing content for prompt: '{prompt[:100]}...' (truncated)" if len(prompt) > 100 else f"Optimizing content for prompt: '{prompt}'")
    
    # If training examples are provided, optimize the generator first
    if args.train:
        training_examples = []
        for train_file in args.train:
            with open(train_file, 'r', encoding='utf-8') as f:
                training_examples.append(f.read())
        
        print(f"Training optimizer with {len(training_examples)} examples...")
        optimizer.optimize(training_examples)
    
    # Generate optimized content
    comparison = optimizer.compare_performance(prompt)
    
    # Print the results
    print("\nStandard Generation:")
    print(f"  Content: {comparison['standard']['content']}")
    print(f"  Score: {comparison['standard']['score']:.4f}")
    
    print("\nOptimized Generation:")
    print(f"  Content: {comparison['optimized']['content']}")
    print(f"  Score: {comparison['optimized']['score']:.4f}")
    
    print("\nImprovement:")
    print(f"  Absolute: {comparison['improvement']['absolute']:.4f}")
    print(f"  Percentage: {comparison['improvement']['percentage']:.2f}%")


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description='LM Abstractions Command Line Interface')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Score command
    score_parser = subparsers.add_parser('score', help='Score text based on attributes')
    score_parser.add_argument('--text', '-t', type=str, help='Text to score')
    score_parser.add_argument('--file', '-f', type=str, help='File containing text to score')
    score_parser.add_argument('--model', '-m', type=str, default='deepseek/deepseek-chat', help='Model to use')
    score_parser.add_argument('--positive-attrs', '-p', type=str, help='Comma-separated list of positive attributes')
    score_parser.add_argument('--negative-attrs', '-n', type=str, help='Comma-separated list of negative attributes')
    score_parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate content using a language model')
    generate_parser.add_argument('--prompt', '-p', type=str, help='Prompt for generation')
    generate_parser.add_argument('--file', '-f', type=str, help='File containing prompt for generation')
    generate_parser.add_argument('--model', '-m', type=str, default='deepseek/deepseek-chat', help='Model to use')
    generate_parser.add_argument('--temperature', '-t', type=float, default=1.0, help='Sampling temperature')
    generate_parser.add_argument('--max-tokens', type=int, default=512, help='Maximum number of tokens to generate')
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Optimize content generation based on scoring metrics')
    optimize_parser.add_argument('--prompt', '-p', type=str, help='Prompt for generation')
    optimize_parser.add_argument('--file', '-f', type=str, help='File containing prompt for generation')
    optimize_parser.add_argument('--model', '-m', type=str, default='deepseek/deepseek-chat', help='Model to use')
    optimize_parser.add_argument('--positive-attrs', '-pos', type=str, help='Comma-separated list of positive attributes')
    optimize_parser.add_argument('--negative-attrs', '-neg', type=str, help='Comma-separated list of negative attributes')
    optimize_parser.add_argument('--method', type=str, default='simple', choices=['simple', 'mipro'], help='Optimization method')
    optimize_parser.add_argument('--candidates', '-c', type=int, default=5, help='Number of candidates to generate')
    optimize_parser.add_argument('--train', '-t', type=str, nargs='+', help='Training example files')
    
    args = parser.parse_args()
    
    if args.command == 'score':
        if not args.text and not args.file:
            score_parser.error('either --text or --file is required')
        score_text(args)
    elif args.command == 'generate':
        if not args.prompt and not args.file:
            generate_parser.error('either --prompt or --file is required')
        generate_content(args)
    elif args.command == 'optimize':
        if not args.prompt and not args.file:
            optimize_parser.error('either --prompt or --file is required')
        optimize_content(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
