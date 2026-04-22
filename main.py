import numpy as np
from loguru import logger
from utils.data_loader import DataLoader
from models.pca_model import PCAModel
from models.svd_model import SVDModel
from utils.clustering import ClusteringPipeline
from utils.visualizer import Visualizer

def rebin_quality(y: np.ndarray) -> np.ndarray:
    """Group quality into 3 tiers: Poor (0-4), Average (5-6), Excellent (7-10)"""
    return np.where(y <= 4, 0, np.where(y <= 6, 1, 2))

def determine_optimal_k(X, method='pca', threshold=0.95):
    """Return smallest number of components that explain at least `threshold` variance."""
    max_comp = X.shape[1]
    if method == 'pca':
        model = PCAModel(max_comp)
        model.fit(X)
        var_ratio = model.get_explained_variance_ratio()
    else:  # svd
        model = SVDModel(max_comp)
        model.fit(X)
        var_ratio = model.get_explained_variance_ratio()
    cumsum = np.cumsum(var_ratio)
    k = np.argmax(cumsum >= threshold) + 1
    logger.info(f"{method.upper()} needs {k} components to explain {threshold*100:.0f}% variance (actual: {cumsum[k-1]*100:.1f}%)")
    return k

def print_component_loadings(pca_model, feature_names, n_components=None):
    """Print top contributing original features for each principal component."""
    if n_components is None:
        n_components = pca_model.components.shape[1]
    components = pca_model.components[:, :n_components]  # shape (n_features, n_components)
    print("\n--- PCA Component Loadings (top 3 features per component) ---")
    for i in range(n_components):
        loadings = components[:, i]
        # Get indices of top 3 absolute loadings
        top_idx = np.argsort(np.abs(loadings))[-3:][::-1]
        top_features = [f"{feature_names[idx]} ({loadings[idx]:.3f})" for idx in top_idx]
        print(f"PC{i+1}: {', '.join(top_features)}")


def process_wine_data(file_path: str, viz: Visualizer):
    logger.info(f"--- Processing Dataset: {file_path} ---")
    
    # 1. Load and preprocess
    loader = DataLoader(file_path)
    loader.load_data()
    loader.clean_data()
    X, y = loader.split_features_target(target_column="quality")
    X_scaled = loader.standardize_features()

    # 2. Quality binning
    y_binned = rebin_quality(y)
    n_clusters = 3

    # 3. Determine optimal number of components (95% variance)
    k_pca = determine_optimal_k(X_scaled, 'pca', threshold=0.95)
    k_svd = determine_optimal_k(X_scaled, 'svd', threshold=0.95)
    feature_names = loader.data.drop(columns=['quality']).columns.tolist()
    # 4. Fit PCA and SVD with optimal components
    pca = PCAModel(n_components=k_pca)
    pca.fit(X_scaled)
    print_component_loadings(pca, feature_names, n_components=k_pca)
    X_pca = pca.transform(X_scaled)
    
    svd = SVDModel(n_components=k_svd)
    X_svd = svd.fit_transform(X_scaled)   # fit_transform centers and projects

    # 5. Clustering on reduced dimensions
    clusterer = ClusteringPipeline(n_clusters=n_clusters)
    pca_labels = clusterer.run_clustering(X_pca)
    svd_labels = clusterer.run_clustering(X_svd)
    
    # 6. Evaluation
    pca_eval = clusterer.evaluate(X_pca, pca_labels)
    svd_eval = clusterer.evaluate(X_svd, svd_labels)
    logger.info(f"PCA variance ratios (first {k_pca}): {pca.get_explained_variance_ratio()}")
    logger.info(f"SVD variance ratios (first {k_svd}): {svd.get_explained_variance_ratio()}")
    logger.info(f"PCA clustering scores: {pca_eval}")
    logger.info(f"SVD clustering scores: {svd_eval}")

    # 7. Visualizations
    file_tag = "Red" if "red" in file_path.lower() else "White"
    viz.plot_feature_loadings(pca.components, feature_names, title=f"{file_tag} Wine PCA Loadings")
    # Cumulative variance plots
    viz.plot_cumulative_variance(pca.get_explained_variance_ratio(), f"{file_tag} Wine PCA Cumulative Variance")
    viz.plot_cumulative_variance(svd.get_explained_variance_ratio(), f"{file_tag} Wine SVD Cumulative Variance")
    
    # Side-by-side variance comparison (first 10 components or min of both)
    max_comp = min(k_pca, k_svd, 10)
    viz.compare_variance_side_by_side(
        pca.get_explained_variance_ratio()[:max_comp],
        svd.get_explained_variance_ratio()[:max_comp]
    )
    
    # 2D and 3D clusters for PCA
    viz.plot_clusters_2d(X_pca, pca_labels, f"2D {file_tag} PCA Clusters (k={k_pca})")
    viz.plot_clusters_3d(X_pca, pca_labels, f"3D {file_tag} PCA Clusters (k={k_pca})")
    
    # 2D and 3D clusters for SVD
    viz.plot_clusters_2d(X_svd, svd_labels, f"2D {file_tag} SVD Clusters (k={k_svd})")
    viz.plot_clusters_3d(X_svd, svd_labels, f"3D {file_tag} SVD Clusters (k={k_svd})")
    
    viz.plot_tsne(X_pca, pca_labels, f"{file_tag} t-SNE on PCA-reduced data")
    viz.plot_tsne(X_scaled, y_binned, f"{file_tag} t-SNE on Original Data")

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