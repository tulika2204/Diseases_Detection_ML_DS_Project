import unittest
import pandas as pd
from data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    """Tests for the DataLoader class."""

    def setUp(self):
        """Set up the DataLoader instance before running tests."""
        self.loader = DataLoader(config_path='./config.yaml')

    def test_load_train_data(self):
        """Test if training data loads correctly."""
        features, labels, df = self.loader.load_dataset('training_data_path')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)  # Ensure data is not empty

    def test_load_test_data(self):
        """Test if test data loads correctly."""
        features, labels, df = self.loader.load_dataset('test_data_path')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

if __name__ == "__main__":
    unittest.main()
