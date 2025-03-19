from data_loader import DataLoader
from model import DiseaseModel
from visualization import plot_feature_correlation
from utils import load_config

if __name__ == "__main__":
    # Load Config
    config = load_config()

    # Load Data
    data_loader = DataLoader(config_path='./config.yaml')
    train_data = data_loader.load_dataset('training_data_path')
    test_data = data_loader.load_dataset('test_data_path')

    # Feature Correlation
    plot_feature_correlation(train_data[2])

    # Train & Evaluate Model
    model = DiseaseModel(model_name='decision_tree', config=config, train_data=train_data, test_data=test_data)
    model.train_model()

    # Make Predictions
    test_accuracy, test_report = model.make_prediction()
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Classification Report:\n{test_report}")
