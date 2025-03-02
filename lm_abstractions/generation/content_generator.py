"""
Content Generator
---------------

Generate content using language models with customizable signatures.
"""

import dspy
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass


class GenerationSignature(dspy.Signature):
    """
    Default generation signature with input prompt and output content.
    """
    prompt = dspy.InputField(type=str, description="Input prompt for content generation.")
    reasoning = dspy.OutputField(type=str, description="The reasoning behind the generated content.")
    content = dspy.OutputField(type=str, description="The generated content.")


class ContentGenerator:
    """
    Generate content using language models with customizable signatures.
    """
    
    def __init__(self, model: str = 'deepseek/deepseek-chat', 
                 signature: dspy.Signature = None,
                 max_tokens: int = 512,
                 temperature: float = 1.0,
                 top_logprobs: int = 10,
                 cache: bool = True):
        """
        Initialize the ContentGenerator.
        
        Args:
            model: Model identifier to use
            signature: Optional custom dspy signature for generation
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
        self.signature = signature or GenerationSignature
        self.generator = dspy.Predict(self.signature)
        
    def generate(self, prompt: str, **kwargs) -> Any:
        """
        Generate content based on a prompt.
        
        Args:
            prompt: Input prompt for generation
            **kwargs: Additional keyword arguments for the generator
            
        Returns:
            dspy.Prediction object with generated content
        """
        # Combine prompt with kwargs for full input
        full_input = {"prompt": prompt, **kwargs}
        
        # Filter kwargs to only include those in the signature's input fields
        input_field_names = [field.name for field in self.signature.input_fields]
        filtered_input = {k: v for k, v in full_input.items() if k in input_field_names}
        
        # Generate content
        result = self.generator(**filtered_input)
        
        return result
    
    def generate_with_context(self, prompt: str, context: str, **kwargs) -> Any:
        """
        Generate content with additional context.
        
        Args:
            prompt: Input prompt for generation
            context: Additional context to guide generation
            **kwargs: Additional keyword arguments for the generator
            
        Returns:
            dspy.Prediction object with generated content
        """
        enhanced_prompt = f"{context}\n\n{prompt}"
        return self.generate(enhanced_prompt, **kwargs)
    
    def generate_variants(self, prompt: str, num_variants: int = 3, **kwargs) -> List[Any]:
        """
        Generate multiple variants of content for the same prompt.
        
        Args:
            prompt: Input prompt for generation
            num_variants: Number of variants to generate
            **kwargs: Additional keyword arguments for the generator
            
        Returns:
            List of dspy.Prediction objects with generated content variants
        """
        variants = []
        
        for _ in range(num_variants):
            variant = self.generate(prompt, **kwargs)
            variants.append(variant)
            
        return variants
    
    def create_specialized_generator(self, 
                                     category: str, 
                                     instructions: str,
                                     max_length: Optional[int] = None) -> 'ContentGenerator':
        """
        Create a specialized generator for a specific content category.
        
        Args:
            category: Category of content (e.g., "tweet", "blog post")
            instructions: Specific instructions for this content type
            max_length: Optional maximum length constraint
            
        Returns:
            New ContentGenerator instance with specialized context
        """
        context = f"""Generate {category} content following these instructions:
        
{instructions}"""
        
        if max_length:
            context += f"\n\nThe content must be under {max_length} characters or tokens."
        
        # Create a closure function that applies this context
        def specialized_generate(prompt, **kwargs):
            return self.generate_with_context(prompt, context, **kwargs)
        
        # Create a new generator with the same settings
        specialized = ContentGenerator(
            model=self.lm.model,
            signature=self.signature,
            max_tokens=self.lm.max_tokens,
            temperature=self.lm.temperature,
            top_logprobs=self.lm.top_logprobs,
            cache=self.lm.cache
        )
        
        # Replace the generate method with our specialized version
        specialized.specialized_generate = specialized_generate
        
        return specialized
