"""
Attribute Scorer
--------------

Score text based on attributes using logprobs from language models.
"""

import dspy
import math
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass


@dataclass
class AttributeScore:
    """
    Container for attribute scoring details.
    """
    attribute: str
    score: float
    true_probs: Dict[str, float]
    
    def __str__(self):
        return f"{self.attribute}: {self.score:.4f}"
    
    def __repr__(self):
        return f"AttributeScore({self.attribute}, {self.score:.4f})"


class TextAttributeScorer:
    """
    Score text based on positive and negative attributes using LM logprobs.
    """
    
    def __init__(self, model: str = 'deepseek/deepseek-chat', 
                 positive_attrs: List[str] = None, 
                 negative_attrs: Optional[List[str]] = None, 
                 temperature: float = 1.0,
                 max_tokens: int = 512,
                 top_logprobs: int = 10,
                 cache: bool = True):
        """
        Initialize the TextAttributeScorer.
        
        Args:
            model: Model identifier to use
            positive_attrs: List of positive attributes to score for
            negative_attrs: List of negative attributes to score for
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            top_logprobs: Number of top logprobs to return per token
            cache: Whether to cache LM responses
        """
        self.lm = dspy.LM(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            cache=cache,
            logprobs=True,
            top_logprobs=top_logprobs
        )
        
        self.positive_attrs = positive_attrs or ["concise", "clear", "informative"]
        self.negative_attrs = negative_attrs or ["verbose", "unclear", "misleading"]
        self.all_attrs = self.positive_attrs + self.negative_attrs
        self.signature = self._construct_signature(self.all_attrs)
        self.temperature = temperature
        
    def _construct_signature(self, attributes: List[str]) -> str:
        """
        Construct a dspy signature for attribute scoring.
        
        Args:
            attributes: List of attributes to include in the signature
            
        Returns:
            dspy signature string
        """
        bool_fields = [f"{attr}_bool: bool" for attr in attributes]
        return f"text -> {', '.join(bool_fields)}"
    
    def _logprob_to_prob(self, logprob: float) -> float:
        """Convert log probability to probability."""
        return math.exp(logprob)
    
    def _extract_bool_probs(self, token_data: Dict) -> Dict[str, float]:
        """
        Extract probability values for boolean tokens.
        
        Args:
            token_data: Token data from logprobs
            
        Returns:
            Dictionary of boolean token probabilities
        """
        bool_probs = {'true': 0.0, 'True': 0.0, 'false': 0.0, 'False': 0.0}
        temperature_adjustment_factor = 100.0  # Higher temperature = more randomness
        
        # First get raw logprobs
        for token_info in token_data['top_logprobs']:
            token = token_info['token']
            if token in bool_probs:
                # Divide logprobs by temperature to simulate higher temperature
                adjusted_logprob = token_info['logprob'] / temperature_adjustment_factor
                bool_probs[token] = self._logprob_to_prob(adjusted_logprob)
        
        # Normalize probabilities to sum to 1
        total_prob = sum(bool_probs.values())
        if total_prob > 0:
            for key in bool_probs:
                bool_probs[key] /= total_prob
                
        return bool_probs
    
    def _calculate_attribute_score(self, bool_probs: Dict[str, float]) -> float:
        """
        Calculate attribute score from boolean probabilities.
        
        Args:
            bool_probs: Dictionary of boolean token probabilities
            
        Returns:
            Score value from -1.0 to 1.0
        """
        return (bool_probs['true'] + bool_probs['True'] - 
                bool_probs['false'] - bool_probs['False'])
    
    def _process_attribute_tokens(self, logprobs_content: List[Dict], all_attrs: List[str]) -> List[AttributeScore]:
        """
        Process token logprobs to extract attribute scores.
        
        Args:
            logprobs_content: List of token logprob data
            all_attrs: List of attributes to process
            
        Returns:
            List of AttributeScore objects
        """
        attribute_scores = []
        current_attr_idx = 0
        
        for token in logprobs_content:
            if token['token'] in ['True', 'False', 'true', 'false']:
                if current_attr_idx < len(all_attrs):
                    bool_probs = self._extract_bool_probs(token)
                    score = self._calculate_attribute_score(bool_probs)
                    attribute_scores.append(AttributeScore(
                        attribute=all_attrs[current_attr_idx],
                        true_probs=bool_probs,
                        score=score,
                    ))
                    current_attr_idx += 1
                
        return attribute_scores

    def score_text(self, text: str) -> Tuple[float, Dict]:
        """
        Score text based on attributes using logprobs.
        
        Args:
            text: Text to score
            
        Returns:
            Tuple containing:
              - Final score (float from -1.0 to 1.0)
              - Details dictionary with attribute scores and statistics
        """
        with dspy.context(lm=self.lm):
            judge = dspy.Predict(self.signature, logprobs=True, top_logprobs=10)
            response = judge(text=text)
        
        if not hasattr(response, 'logprobs') or not isinstance(response.logprobs, dict):
            raise ValueError("No logprobs in response")
            
        if 'content' not in response.logprobs:
            raise ValueError("No content in logprobs")
            
        attribute_scores = self._process_attribute_tokens(response.logprobs['content'], self.all_attrs)
        
        pos_scores = attribute_scores[:len(self.positive_attrs)]
        neg_scores = attribute_scores[len(self.positive_attrs):] if self.negative_attrs else []
        
        pos_mean = sum(score.score for score in pos_scores) / len(pos_scores) if pos_scores else 0
        neg_mean = sum(score.score for score in neg_scores) / len(neg_scores) if neg_scores else 0
        
        final_score = (pos_mean - neg_mean) / 2
        
        details = {
            'positive_attributes': {
                attr: score for attr, score in zip(self.positive_attrs, pos_scores)
            },
            'negative_attributes': {
                attr: score for attr, score in zip(self.negative_attrs, neg_scores)
            },
            'positive_mean': pos_mean,
            'negative_mean': neg_mean,
            'final_score': final_score
        }
        
        return final_score, details
