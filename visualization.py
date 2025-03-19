import matplotlib.pyplot as plt
import seaborn as sn

def generate_correlation_heatmap(data_frame, save_path="feature_correlation.png"):
    """
    Generates and saves a feature correlation heatmap.

    Args:
        data_frame (pd.DataFrame): DataFrame containing features.
        save_path (str): Path to save the heatmap image.
    """
    plt.figure(figsize=(10, 8))
    sn.heatmap(data_frame.corr(), cmap="YlGnBu", square=True)
    plt.title("Feature Correlation")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()  # Prevent memory leaks

