import pandas as pd
import yaml

class DataLoader:
    """Handles dataset loading from CSV files."""

    def __init__(self, config_path='./config.yaml'):
        self.config = self._load_config(config_path)

    def _load_config(self, file_path):
        """Load configuration from a YAML file."""
        try:
            with open(file_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading config file: {e}")
            return {}

    def load_dataset(self, dataset_key):
        """Load training or test dataset."""
        try:
            dataset_path = self.config['dataset'][dataset_key]
            df = pd.read_csv(dataset_path)
            feature_columns = df.columns[:-1] if 'test' in dataset_key else df.columns[:-2]
            features = df[feature_columns]
            labels = df['prognosis']
            return features, labels, df
        except Exception as e:
            print(f"Error loading {dataset_key}: {e}")
            return None, None, None
