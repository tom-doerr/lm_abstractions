"""
Content Optimization Example
--------------------------

Demonstrates how to use ContentGenerator and ContentOptimizer to generate and optimize content.
"""

import sys
import os

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lm_abstractions.scoring import TextAttributeScorer
from lm_abstractions.generation import ContentGenerator
from lm_abstractions.optimization import ContentOptimizer

def simple_optimization_example():
    """Example of simple optimization (generating multiple candidates and picking the best)."""
    print("=" * 80)
    print("Simple Optimization Example")
    print("=" * 80)
    
    # Define attributes for scoring
    positive_attrs = ["engaging", "clear", "informative", "relevant", "entertaining"]
    negative_attrs = ["verbose", "unclear", "misleading", "irrelevant", "boring"]
    
    # Initialize the optimizer
    optimizer = ContentOptimizer(
        model="deepseek/deepseek-chat",
        positive_attrs=positive_attrs,
        negative_attrs=negative_attrs,
        optimization_method="simple",
        num_candidates=3
    )
    
    # Generate optimized content
    prompt = "Write a tweet about AI and productivity"
    
    print(f"Generating optimized content for prompt: '{prompt}'")
    print("Comparing standard vs. optimized generation...")
    
    # Compare performance
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

def mipro_optimization_example():
    """Example of optimization using dspy's MIPROv2."""
    print("\n" + "=" * 80)
    print("MIPROv2 Optimization Example")
    print("=" * 80)
    
    # Define attributes for scoring
    positive_attrs = ["engaging", "clear", "informative", "relevant", "entertaining"]
    negative_attrs = ["verbose", "unclear", "misleading", "irrelevant", "boring"]
    
    # Initialize the optimizer
    optimizer = ContentOptimizer(
        model="deepseek/deepseek-chat",
        positive_attrs=positive_attrs,
        negative_attrs=negative_attrs,
        optimization_method="mipro",
        num_candidates=3
    )
    
    # Create training examples
    training_examples = [
        "Write a tweet about AI and creativity",
        "Generate a social media post about machine learning",
        "Create a tweet about data science",
        "Write a social media update about neural networks",
        "Compose a tweet about technology trends"
    ]
    
    print("Training optimization model with examples...")
    optimizer.optimize(training_examples)
    
    # Test prompts for evaluation
    test_prompts = [
        "Write a tweet about AI ethics",
        "Generate a social media post about deep learning"
    ]
    
    print("\nEvaluating optimization performance...")
    results = optimizer.evaluate(test_prompts)
    
    # Print the summary results
    print("\nEvaluation Summary:")
    print(f"  Average Standard Score: {results['summary']['avg_standard_score']:.4f}")
    print(f"  Average Optimized Score: {results['summary']['avg_optimized_score']:.4f}")
    print(f"  Average Improvement (Absolute): {results['summary']['avg_improvement']['absolute']:.4f}")
    print(f"  Average Improvement (Percentage): {results['summary']['avg_improvement']['percentage']:.2f}%")
    
    # Print detailed results for each prompt
    print("\nDetailed Results:")
    for i, result in enumerate(results['comparisons']):
        print(f"\nTest Prompt {i+1}: '{result['prompt']}'")
        print(f"  Standard: '{result['comparison']['standard']['content']}'")
        print(f"  Optimized: '{result['comparison']['optimized']['content']}'")
        print(f"  Improvement: {result['comparison']['improvement']['absolute']:.4f} ({result['comparison']['improvement']['percentage']:.2f}%)")

def main():
    """Run both optimization examples."""
    simple_optimization_example()
    mipro_optimization_example()

if __name__ == "__main__":
    main()
