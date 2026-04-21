import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from loguru import logger
from typing import Optional, List

class Visualizer:
    """
    A professional plotting utility for Spectral Learning results.
    Handles variance curves, 2D/3D projections, and model comparisons.
    """

    def __init__(self, style: str = "whitegrid"):
        sns.set_theme(style=style)
        plt.rcParams["figure.figsize"] = (10, 6)

    def plot_cumulative_variance(self, variance_ratio: np.ndarray, title: str = "Cumulative Explained Variance"):
        """
        Plots the cumulative sum of variance to show how many components 
        are needed to reach a certain threshold (e.g., 95%).
        """
        try:
            cumulative_variance = np.cumsum(variance_ratio)
            plt.figure()
            plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
            plt.axhline(y=0.95, color='r', linestyle=':', label='95% Threshold')
            plt.xlabel("Number of Components")
            plt.ylabel("Cumulative Explained Variance")
            plt.title(title)
            plt.legend()
            plt.show()
            logger.success(f"Plotted cumulative variance for {title}")
        except Exception as e:
            logger.error(f"Failed to plot cumulative variance: {e}")

    def plot_clusters_2d(self, data: np.ndarray, labels: np.ndarray, title: str = "2D Cluster Projection"):
        """Creates a 2D scatter plot colored by cluster labels."""
        if data.shape[1] < 2:
            logger.error("Data must have at least 2 dimensions for 2D plot.")
            return

        plt.figure()
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette="viridis", alpha=0.7)
        plt.title(title)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.show()

    def plot_clusters_3d(self, data: np.ndarray, labels: np.ndarray, title: str = "3D Cluster Projection"):
        """Creates a 3D scatter plot for deeper spatial analysis."""
        if data.shape[1] < 3:
            logger.error("Data must have at least 3 dimensions for 3D plot.")
            return

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='viridis', alpha=0.6)
        
        ax.set_title(title)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_zlabel("PC 3")
        plt.colorbar(scatter)
        plt.show()

    def compare_variance_side_by_side(self, pca_vars: np.ndarray, svd_vars: np.ndarray):
        """Compares individual component variance between PCA and SVD."""
        labels = [f"Comp {i+1}" for i in range(len(pca_vars))]
        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots()
        ax.bar(x - width/2, pca_vars, width, label='PCA (Eigen)')
        ax.bar(x + width/2, svd_vars, width, label='SVD (Singular)')

        ax.set_ylabel('Variance Ratio')
        ax.set_title('Variance Explained: PCA vs SVD')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        plt.show()