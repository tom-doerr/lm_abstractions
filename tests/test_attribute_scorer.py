"""
Test the TextAttributeScorer class.
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lm_abstractions.scoring import TextAttributeScorer, AttributeScore


class MockLogprobs:
    """Mock logprobs response for testing."""
    
    def __init__(self, content):
        self.logprobs = {'content': content}


class TestAttributeScorer(unittest.TestCase):
    """Test the TextAttributeScorer class."""
    
    def setUp(self):
        """Set up the test case."""
        self.positive_attrs = ["concise", "clear"]
        self.negative_attrs = ["verbose", "unclear"]
        
        # Create a mock scorer without actual LM initialization
        with patch('dspy.LM'):
            self.scorer = TextAttributeScorer(
                model="mock-model",
                positive_attrs=self.positive_attrs,
                negative_attrs=self.negative_attrs
            )
    
    def test_construct_signature(self):
        """Test the _construct_signature method."""
        signature = self.scorer._construct_signature(["attr1", "attr2"])
        self.assertEqual(signature, "text -> attr1_bool: bool, attr2_bool: bool")
    
    def test_calculate_attribute_score(self):
        """Test the _calculate_attribute_score method."""
        bool_probs = {
            'true': 0.7,
            'True': 0.1,
            'false': 0.1,
            'False': 0.1
        }
        
        score = self.scorer._calculate_attribute_score(bool_probs)
        self.assertAlmostEqual(score, 0.6)
    
    def test_process_attribute_tokens(self):
        """Test the _process_attribute_tokens method."""
        # Create mock token data
        tokens = [
            {
                'token': 'True',
                'top_logprobs': [
                    {'token': 'True', 'logprob': -0.5},
                    {'token': 'False', 'logprob': -1.5}
                ]
            },
            {
                'token': 'False',
                'top_logprobs': [
                    {'token': 'False', 'logprob': -0.3},
                    {'token': 'True', 'logprob': -2.0}
                ]
            }
        ]
        
        # Patch the _extract_bool_probs method to return consistent values
        with patch.object(
            self.scorer, '_extract_bool_probs', 
            side_effect=[
                {'true': 0.6, 'True': 0.2, 'false': 0.1, 'False': 0.1},
                {'true': 0.1, 'True': 0.1, 'false': 0.3, 'False': 0.5}
            ]
        ):
            scores = self.scorer._process_attribute_tokens(tokens, ["attr1", "attr2"])
            
            self.assertEqual(len(scores), 2)
            self.assertEqual(scores[0].attribute, "attr1")
            self.assertEqual(scores[1].attribute, "attr2")
            self.assertAlmostEqual(scores[0].score, 0.6)
            self.assertAlmostEqual(scores[1].score, -0.6)
    
    @patch('dspy.Predict')
    @patch('dspy.context')
    def test_score_text(self, mock_context, mock_predict):
        """Test the score_text method."""
        # Set up the mock response
        mock_instance = mock_predict.return_value
        
        # Create mock token data
        token_data = [
            {
                'token': 'True',
                'top_logprobs': [
                    {'token': 'True', 'logprob': -0.5},
                    {'token': 'False', 'logprob': -1.5}
                ]
            },
            {
                'token': 'False',
                'top_logprobs': [
                    {'token': 'False', 'logprob': -0.3},
                    {'token': 'True', 'logprob': -2.0}
                ]
            },
            {
                'token': 'True',
                'top_logprobs': [
                    {'token': 'True', 'logprob': -0.2},
                    {'token': 'False', 'logprob': -2.0}
                ]
            },
            {
                'token': 'True',
                'top_logprobs': [
                    {'token': 'True', 'logprob': -0.1},
                    {'token': 'False', 'logprob': -3.0}
                ]
            }
        ]
        
        # Create mock response
        mock_instance.return_value = MockLogprobs(token_data)
        
        # Patch the _extract_bool_probs method to return consistent values
        bool_probs_values = [
            {'true': 0.6, 'True': 0.2, 'false': 0.1, 'False': 0.1},  # concise
            {'true': 0.7, 'True': 0.1, 'false': 0.1, 'False': 0.1},  # clear
            {'true': 0.1, 'True': 0.1, 'false': 0.3, 'False': 0.5},  # verbose
            {'true': 0.2, 'True': 0.1, 'false': 0.3, 'False': 0.4}   # unclear
        ]
        
        with patch.object(
            self.scorer, '_extract_bool_probs', 
            side_effect=bool_probs_values
        ):
            score, details = self.scorer.score_text("Test text")
            
            # Check that the result structure is correct
            self.assertIsInstance(score, float)
            self.assertIsInstance(details, dict)
            
            # Check that the score was calculated correctly
            # pos_mean = (0.6 + 0.6) / 2 = 0.6
            # neg_mean = (-0.6 + -0.4) / 2 = -0.5
            # final_score = (0.6 - -0.5) / 2 = 0.55
            self.assertAlmostEqual(score, 0.55)
            
            # Check that details contain the expected keys
            self.assertIn('positive_attributes', details)
            self.assertIn('negative_attributes', details)
            self.assertIn('positive_mean', details)
            self.assertIn('negative_mean', details)
            self.assertIn('final_score', details)


if __name__ == '__main__':
    unittest.main()
