"""
Text Scoring Example
------------------

Demonstrates how to use the TextAttributeScorer to score text based on attributes.
"""

import sys
import os

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lm_abstractions.scoring import TextAttributeScorer

def main():
    # Define attributes for scoring
    positive_attrs = ["engaging", "clear", "informative", "relevant", "entertaining"]
    negative_attrs = ["verbose", "unclear", "misleading", "irrelevant", "boring"]
    
    # Initialize the scorer with your preferred model
    scorer = TextAttributeScorer(
        model="deepseek/deepseek-chat",
        positive_attrs=positive_attrs,
        negative_attrs=negative_attrs
    )
    
    # Sample text to score
    text = "My LLM API bills increase with every token price drop"
    
    print(f"Scoring text: '{text}'")
    
    # Score the text
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

if __name__ == "__main__":
    main()
