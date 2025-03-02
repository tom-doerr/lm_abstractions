"""
Logprob Extractor
----------------

Utilities for extracting and analyzing logprobs from language model outputs.
"""

import dspy
import math
from typing import Dict, Any, List, Optional, Union, Tuple


class LogprobExtractor:
    """
    Extract and analyze logprobs from language model outputs.
    """
    
    def __init__(self, model: str = 'deepseek/deepseek-chat', 
                 max_tokens: int = 512, 
                 temperature: float = 1.0,
                 top_logprobs: int = 10,
                 cache: bool = True):
        """
        Initialize the LogprobExtractor.
        
        Args:
            model: Model identifier to use
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
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
        dspy.settings.configure(lm=self.lm)
        
    def logprob_to_prob(self, logprob: float) -> float:
        """Convert log probability to probability."""
        return math.exp(logprob)
    
    def extract_logprobs(self, text: str, signature: str = None) -> Dict[str, Any]:
        """
        Extract logprobs for a given text using a specified signature or simple prediction.
        
        Args:
            text: Input text to analyze
            signature: Optional dspy signature string (e.g., 'text -> answer')
            
        Returns:
            Dictionary containing logprob information
        """
        try:
            if signature:
                predictor = dspy.Predict(signature, logprobs=True, top_logprobs=self.lm.top_logprobs)
                response = predictor(text=text)
            else:
                response = self.lm(text, logprobs=True, top_logprobs=self.lm.top_logprobs)
                
            if not hasattr(response, 'logprobs') or not isinstance(response.logprobs, dict):
                raise ValueError("No logprobs in response")
                
            return response.logprobs
        except Exception as e:
            raise RuntimeError(f"Error extracting logprobs: {str(e)}")
            
    def extract_token_probs(self, text: str, tokens_of_interest: List[str]) -> Dict[str, float]:
        """
        Extract probability scores for specific tokens of interest.
        
        Args:
            text: Input text to analyze
            tokens_of_interest: List of tokens to extract probabilities for
            
        Returns:
            Dictionary mapping tokens to their probability scores
        """
        logprobs = self.extract_logprobs(text)
        
        if 'content' not in logprobs:
            raise ValueError("No content in logprobs response")
            
        token_probs = {}
        
        for token_data in logprobs['content']:
            for top_token in token_data.get('top_logprobs', []):
                if top_token['token'] in tokens_of_interest:
                    token_probs[top_token['token']] = self.logprob_to_prob(top_token['logprob'])
                    
        return token_probs
    
    def extract_boolean_probs(self, text: str, signature: str) -> Dict[str, Dict[str, float]]:
        """
        Extract probabilities for boolean outputs (True/False).
        
        Args:
            text: Input text to analyze
            signature: Signature containing boolean outputs (e.g., 'text -> is_good: bool')
            
        Returns:
            Dictionary mapping boolean field names to their True/False probabilities
        """
        logprobs = self.extract_logprobs(text, signature)
        
        if 'content' not in logprobs:
            raise ValueError("No content in logprobs response")
            
        bool_probs = {}
        current_field = None
        
        # Analyze the tokens to identify bool fields and their probabilities
        for token in logprobs['content']:
            # First check if this token is a boolean value
            if token['token'] in ['True', 'False', 'true', 'false']:
                if current_field:
                    field_probs = {'true': 0.0, 'false': 0.0}
                    
                    # Extract probabilities for boolean tokens
                    for token_info in token.get('top_logprobs', []):
                        if token_info['token'].lower() == 'true':
                            field_probs['true'] += self.logprob_to_prob(token_info['logprob'])
                        elif token_info['token'].lower() == 'false':
                            field_probs['false'] += self.logprob_to_prob(token_info['logprob'])
                            
                    # Normalize probabilities
                    total_prob = field_probs['true'] + field_probs['false']
                    if total_prob > 0:
                        field_probs['true'] /= total_prob
                        field_probs['false'] /= total_prob
                        
                    bool_probs[current_field] = field_probs
                    current_field = None
            
            # Check if this token indicates a field name
            elif '_bool' in token['token']:
                # Extract field name (removing _bool suffix)
                current_field = token['token'].replace('_bool', '')
                
        return bool_probs
