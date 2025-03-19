import matplotlib.pyplot as plt
import seaborn as sn

def plot_feature_correlation(data_frame, save_path='feature_correlation.png', show_fig=False):
    """Plots and saves feature correlation heatmap."""
    plt.figure(figsize=(10, 8))
    sn.heatmap(data_frame.corr(), cmap="YlGnBu", square=True, annot=False)
    plt.title("Feature Correlation")
    plt.tight_layout()
    plt.savefig(save_path)
    if show_fig:
        plt.show()
