import unittest
from utils import load_config

class TestUtils(unittest.TestCase):
    """Tests for utility functions."""

    def test_load_config(self):
        """Test if config loads correctly."""
        config = load_config('./config.yaml')
        self.assertIn('dataset', config)  # Ensure dataset key exists
        self.assertIn('model', config)  # Ensure model key exists

if __name__ == "__main__":
    unittest.main()
