"""
LM Abstractions
---------------

A Python library for working with language models, focused on logprob-based scoring,
content generation, and optimization.
"""

from lm_abstractions.scoring import TextAttributeScorer, LogprobExtractor
from lm_abstractions.generation import ContentGenerator
from lm_abstractions.optimization import ContentOptimizer

__version__ = '0.1.0'

# Command-line interface
from lm_abstractions.cli import main as cli_main
