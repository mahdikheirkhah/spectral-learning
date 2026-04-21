import numpy as np
from loguru import logger
from utils.data_loader import DataLoader
from models.pca_model import PCAModel
from models.svd_model import SVDModel
from utils.clustering import ClusteringPipeline
from utils.visualizer import Visualizer

def rebin_quality(y: np.ndarray) -> np.ndarray:
    """
    Groups quality scores into 3 tiers:
    Poor (0-4) -> 0
    Average (5-6) -> 1
    Excellent (7-10) -> 2
    """
    return np.where(y <= 4, 0, np.where(y <= 6, 1, 2))

def process_wine_data(file_path: str, viz: Visualizer):
    logger.info(f"--- Processing Dataset: {file_path} ---")
    
    # 1. Load and Standardize
    loader = DataLoader(file_path)
    loader.load_data()
    loader.clean_data()
    X, y = loader.split_features_target(target_column="quality")
    X_scaled = loader.standardize_features()

    # 2. Distribution Analysis (Before Binning)
    unique_qualities, counts = np.unique(y, return_counts=True)
    print(f"\nDistribution Analysis for {file_path}")
    print("-" * 40)
    for q, c in zip(unique_qualities, counts):
        print(f"Quality {int(q)}: {c} ({c/len(y)*100:.2f}%)")
    
    # 3. Apply Re-binning Logic
    y_binned = rebin_quality(y)
    n_clusters = 3  # Poor, Average, Excellent
    
    # 4. Spectral Decomposition (n_components=3)
    pca = PCAModel(n_components=3)
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)

    svd = SVDModel(n_components=3)
    X_svd = svd.fit_transform(X_scaled)

    # 5. Clustering on PCA Space
    clusterer = ClusteringPipeline(n_clusters=n_clusters)
    pca_labels = clusterer.run_clustering(X_pca)
    
    # 6. Evaluation & Logging
    pca_eval = clusterer.evaluate(X_pca, pca_labels)
    logger.info(f"Variance Ratios: {pca.get_explained_variance_ratio()}")
    logger.info(f"Clustering Scores: {pca_eval}")

    # 7. Visualizations
    file_tag = "Red" if "red" in file_path.lower() else "White"
    viz.plot_cumulative_variance(pca.get_explained_variance_ratio(), f"{file_tag} Wine PCA Convergence")
    viz.plot_clusters_2d(X_pca, pca_labels, f"2D {file_tag} Wine Clusters (3 Tiers)")
    viz.plot_clusters_3d(X_pca, pca_labels, f"3D {file_tag} Wine Spatial Separation")

def main():
    datasets = ["data/winequality-red.csv", "data/winequality-white.csv"]
    viz = Visualizer()
    
    for data_file in datasets:
        try:
            process_wine_data(data_file, viz)
        except Exception as e:
            logger.error(f"Failed to process {data_file}: {e}")

if __name__ == "__main__":
    main()