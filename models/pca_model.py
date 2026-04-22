import numpy as np
from loguru import logger
from typing import Optional
from utils.matrix_operations import center_data, compute_covariance_matrix, compute_top_eigenvectors

class PCAModel:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components: Optional[np.ndarray] = None
        self.mean: Optional[np.ndarray] = None
        self.explained_variance: Optional[np.ndarray] = None
        self.total_global_variance: Optional[float] = None

    def fit(self, X: np.ndarray) -> None:
        try:
            if self.n_components > X.shape[1]:
                raise ValueError(f"n_components ({self.n_components}) > features ({X.shape[1]})")
            logger.info(f"Fitting PCA for {self.n_components} components.")
            x_centered, self.mean = center_data(X)
            cov_matrix = compute_covariance_matrix(x_centered)

            # Get all eigenvalues & eigenvectors sorted
            all_eigenvals, all_eigenvecs = compute_top_eigenvectors(cov_matrix, cov_matrix.shape[0])
            self.total_global_variance = np.sum(all_eigenvals)

            # Keep top k
            self.explained_variance = all_eigenvals[:self.n_components]
            self.components = all_eigenvecs[:, :self.n_components]

            logger.success("PCA fitting successful.")
        except Exception as e:
            logger.error(f"PCA fitting failed: {e}")
            raise

    def get_explained_variance_ratio(self) -> np.ndarray:
        if self.explained_variance is None:
            raise ValueError("Model must be fitted.")
        return self.explained_variance / self.total_global_variance

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.components is None or self.mean is None:
            raise ValueError("Model not fitted.")
        return np.dot(X - self.mean, self.components)