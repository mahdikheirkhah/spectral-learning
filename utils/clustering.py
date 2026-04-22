from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from loguru import logger
import numpy as np
from typing import Dict

class ClusteringPipeline:
    """
    Handles clustering operations and evaluation metrics for reduced data.
    """

    def __init__(self, n_clusters: int = 3):
        self.model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        self.n_clusters = n_clusters

    def run_clustering(self, data: np.ndarray) -> np.ndarray:
        """Applies K-means and returns cluster labels."""
        try:
            logger.info(f"Running K-Means clustering for {self.n_clusters} clusters.")
            labels = self.model.fit_predict(data)
            return labels
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            raise

    def evaluate(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Calculates separability scores.
        Silhouette Score: 1.0 is best (well separated), -1.0 is worst.
        """
        try:
            if len(np.unique(labels)) < 2:
                logger.warning("Less than 2 clusters found, silhouette score not defined.")
                return {"silhouette": np.nan, "calinski_harabasz": np.nan}
            scores = {
                "silhouette": silhouette_score(data, labels),
                "calinski_harabasz": calinski_harabasz_score(data, labels)
            }
            logger.success(f"Evaluation scores: {scores}")
            return scores
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise