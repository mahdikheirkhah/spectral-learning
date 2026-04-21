from utils.data_loader import DataLoader
from models.pca_model import PCAModel
from models.svd_model import SVDModel
from utils.clustering import ClusteringPipeline
from utils.visualizer import Visualizer
import numpy as np
from loguru import logger

def main():
    # 1. Pipeline: Load & Preprocess
    loader = DataLoader("data/winequality-white.csv")
    loader.load_data()
    loader.clean_data()
    X, y = loader.split_features_target(target_column="quality")
    X_scaled = loader.standardize_features()

    # Determine clusters based on your log finding
    unique_qualities = np.unique(y)
    n_clusters = len(unique_qualities)
    logger.info(f"Detected {n_clusters} unique quality levels: {unique_qualities}")

    # 2. PCA & SVD Execution (n_components=3 for the 3D plot requirement)
    pca = PCAModel(n_components=3)
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)

    svd = SVDModel(n_components=3)
    X_svd = svd.fit_transform(X_scaled)

    # 3. Clustering
    clusterer = ClusteringPipeline(n_clusters=n_clusters)
    pca_labels = clusterer.run_clustering(X_pca)
    svd_labels = clusterer.run_clustering(X_svd)

    # 4. Phase 5 Visualizations
    viz = Visualizer()
    logger.info(f"Explained variance ratio (PCA): {pca.get_explained_variance_ratio()}")
    logger.info(f"Explained variance ratio (SVD): {svd.get_explained_variance_ratio()}")
    # Cumulative Variance (Convergence)
    viz.plot_cumulative_variance(pca.get_explained_variance_ratio(), "PCA Convergence")
    
    # Side-by-Side Comparison (Proving the SVD/PCA identity)
    viz.compare_variance_side_by_side(pca.get_explained_variance_ratio(), 
                                      svd.get_explained_variance_ratio())

    # 2D and 3D Projections
    viz.plot_clusters_2d(X_pca, pca_labels, "2D PCA Cluster Map")
    viz.plot_clusters_3d(X_pca, pca_labels, "3D PCA Spatial Separation")

if __name__ == "__main__":
    main()