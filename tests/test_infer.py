import unittest
import joblib
import pandas as pd

class TestInference(unittest.TestCase):
    """Tests for the inference module."""

    def setUp(self):
        """Load a trained model and sample test data."""
        self.model = joblib.load("./trained_data/decision_tree.joblib")
        self.test_data = pd.read_csv("./dataset/test_data.csv").iloc[:1, :-1]  # Use one test sample

    def test_model_prediction(self):
        """Test if model makes predictions."""
        prediction = self.model.predict(self.test_data)
        self.assertIsNotNone(prediction)

if __name__ == "__main__":
    unittest.main()
