"""
Content Optimizer
---------------

Optimize content generation using scoring metrics.
"""

import dspy
from typing import List, Dict, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass
import time

from lm_abstractions.scoring.attribute_scorer import TextAttributeScorer
from lm_abstractions.generation.content_generator import ContentGenerator


class ContentOptimizer:
    """
    Optimize content generation using scoring metrics.
    """
    
    def __init__(self, 
                 scorer: Optional[TextAttributeScorer] = None,
                 generator: Optional[ContentGenerator] = None,
                 model: str = 'deepseek/deepseek-chat',
                 positive_attrs: Optional[List[str]] = None,
                 negative_attrs: Optional[List[str]] = None,
                 optimization_method: str = "mipro",
                 num_candidates: int = 5):
        """
        Initialize ContentOptimizer.
        
        Args:
            scorer: TextAttributeScorer instance
            generator: ContentGenerator instance
            model: Model identifier (used if scorer or generator not provided)
            positive_attrs: List of positive attributes for scoring
            negative_attrs: List of negative attributes for scoring
            optimization_method: Optimization method ("mipro" or "simple")
            num_candidates: Number of candidates to generate for simple optimization
        """
        # Create scorer if not provided
        if scorer is None:
            positive_attrs = positive_attrs or ["concise", "clear", "informative", "relevant", "engaging"]
            negative_attrs = negative_attrs or ["verbose", "unclear", "misleading", "irrelevant", "boring"]
            self.scorer = TextAttributeScorer(model, positive_attrs, negative_attrs)
        else:
            self.scorer = scorer
            
        # Create generator if not provided
        if generator is None:
            self.generator = ContentGenerator(model)
        else:
            self.generator = generator
            
        self.optimization_method = optimization_method
        self.num_candidates = num_candidates
        self.optimized_program = None
    
    def _score_content(self, example, pred, trace=None) -> float:
        """
        Score generated content using the attribute scorer.
        
        Args:
            example: Input example
            pred: Generated prediction
            trace: Optional trace data
            
        Returns:
            Content score
        """
        # Access the content field, which may have different names in different signatures
        content_fields = ['content', 'tweet_text', 'response', 'output', 'text']
        content = None
        
        for field in content_fields:
            if hasattr(pred, field):
                content = getattr(pred, field)
                break
                
        if content is None:
            raise ValueError("Cannot find content field in prediction")
            
        score, _ = self.scorer.score_text(content)
        return score
    
    def _simple_optimize(self, prompt: str, num_candidates: int = 5) -> Any:
        """
        Simple optimization by generating multiple candidates and selecting the best.
        
        Args:
            prompt: Input prompt
            num_candidates: Number of candidates to generate
            
        Returns:
            Best candidate
        """
        candidates = []
        scores = []
        
        for _ in range(num_candidates):
            candidate = self.generator.generate(prompt)
            score, _ = self._score_content(None, candidate)
            candidates.append(candidate)
            scores.append(score)
            
        # Select candidate with highest score
        best_idx = scores.index(max(scores))
        return candidates[best_idx]
    
    def _mipro_optimize(self, trainset) -> dspy.Module:
        """
        Optimize using dspy's MIPROv2.
        
        Args:
            trainset: Training examples
            
        Returns:
            Optimized dspy program
        """
        # Initialize MIPROv2 optimizer
        teleprompter = dspy.teleprompt.MIPROv2(
            metric=self._score_content,
            num_candidates=self.num_candidates,
            max_bootstrapped_demos=2,
            max_labeled_demos=2,
            num_threads=1,
            max_errors=100,
        )
        
        # Optimize the program
        optimized_program = teleprompter.compile(
            self.generator,
            trainset=trainset,
            minibatch_size=8,
            requires_permission_to_run=False,
        )
        
        self.optimized_program = optimized_program
        return optimized_program
    
    def optimize(self, training_examples: List[str]) -> Any:
        """
        Optimize content generation using the selected method.
        
        Args:
            training_examples: List of training prompts or examples
            
        Returns:
            Optimized program or None
        """
        # Convert string examples to dspy.Example objects if needed
        trainset = []
        
        for example in training_examples:
            if isinstance(example, str):
                # Determine input field name from generator signature
                input_field = self.generator.signature.input_fields[0].name
                trainset.append(dspy.Example(**{input_field: example}).with_inputs(input_field))
            else:
                trainset.append(example)
        
        if self.optimization_method == "mipro":
            return self._mipro_optimize(trainset)
        else:
            # Simple optimization doesn't return a program, so we don't modify self.optimized_program
            return None
    
    def generate_optimized(self, prompt: str, **kwargs) -> Any:
        """
        Generate optimized content for a prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional keyword arguments for generation
            
        Returns:
            Optimized content
        """
        if self.optimization_method == "simple" or not self.optimized_program:
            return self._simple_optimize(prompt, self.num_candidates)
        else:
            # Determine input field name from generator signature
            input_field = self.generator.signature.input_fields[0].name
            
            # Use the optimized program
            input_dict = {input_field: prompt, **kwargs}
            return self.optimized_program(**input_dict)
    
    def compare_performance(self, prompt: str) -> Dict:
        """
        Compare performance between standard and optimized generation.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Dictionary with comparison results
        """
        # Generate standard content
        standard_result = self.generator.generate(prompt)
        standard_content = None
        
        # Try to extract content from various possible field names
        for field in ['content', 'tweet_text', 'response', 'output', 'text']:
            if hasattr(standard_result, field):
                standard_content = getattr(standard_result, field)
                break
                
        if standard_content is None:
            raise ValueError("Cannot extract content from standard generation")
            
        standard_score, standard_details = self.scorer.score_text(standard_content)
        
        # Generate optimized content
        optimized_result = self.generate_optimized(prompt)
        optimized_content = None
        
        # Try to extract content from various possible field names
        for field in ['content', 'tweet_text', 'response', 'output', 'text']:
            if hasattr(optimized_result, field):
                optimized_content = getattr(optimized_result, field)
                break
                
        if optimized_content is None:
            raise ValueError("Cannot extract content from optimized generation")
            
        optimized_score, optimized_details = self.scorer.score_text(optimized_content)
        
        # Calculate improvement
        score_improvement = optimized_score - standard_score
        percentage_improvement = (score_improvement / abs(standard_score)) * 100 if standard_score != 0 else float('inf')
        
        return {
            'standard': {
                'content': standard_content,
                'score': standard_score,
                'details': standard_details
            },
            'optimized': {
                'content': optimized_content,
                'score': optimized_score,
                'details': optimized_details
            },
            'improvement': {
                'absolute': score_improvement,
                'percentage': percentage_improvement
            }
        }
        
    def evaluate(self, test_prompts: List[str]) -> Dict:
        """
        Evaluate optimizer performance on multiple test prompts.
        
        Args:
            test_prompts: List of test prompts
            
        Returns:
            Dictionary with evaluation results
        """
        results = {
            'comparisons': [],
            'summary': {}
        }
        
        standard_scores = []
        optimized_scores = []
        
        for prompt in test_prompts:
            comparison = self.compare_performance(prompt)
            results['comparisons'].append({
                'prompt': prompt,
                'comparison': comparison
            })
            
            standard_scores.append(comparison['standard']['score'])
            optimized_scores.append(comparison['optimized']['score'])
        
        # Calculate summary statistics
        avg_standard = sum(standard_scores) / len(standard_scores) if standard_scores else 0
        avg_optimized = sum(optimized_scores) / len(optimized_scores) if optimized_scores else 0
        avg_improvement = avg_optimized - avg_standard
        avg_percentage = (avg_improvement / abs(avg_standard)) * 100 if avg_standard != 0 else float('inf')
        
        results['summary'] = {
            'avg_standard_score': avg_standard,
            'avg_optimized_score': avg_optimized,
            'avg_improvement': {
                'absolute': avg_improvement,
                'percentage': avg_percentage
            },
            'num_prompts': len(test_prompts)
        }
        
        return results
