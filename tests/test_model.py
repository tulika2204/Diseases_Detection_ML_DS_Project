import unittest
from model import DiseaseModel
from data_loader import DataLoader
from config import load_config

class TestModel(unittest.TestCase):
    """Tests for the DiseaseModel class."""

    def setUp(self):
        """Set up the model with training data."""
        config = load_config('./config.yaml')
        data_loader = DataLoader(config_path='./config.yaml')
        train_data = data_loader.load_dataset('training_data_path')
        test_data = data_loader.load_dataset('test_data_path')

        self.model = DiseaseModel(model_name="decision_tree", config=config, train_data=train_data, test_data=test_data)

    def test_model_initialization(self):
        """Test if the model initializes correctly."""
        self.assertIsNotNone(self.model._get_model())

    def test_model_training(self):
        """Test if the model trains without errors."""
        try:
            self.model.train_model()
            trained = True
        except Exception:
            trained = False
        self.assertTrue(trained)

if __name__ == "__main__":
    unittest.main()
