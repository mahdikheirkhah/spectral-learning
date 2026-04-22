import numpy as np
from loguru import logger
from typing import Optional

class SVDModel:
    """Dimensionality reduction using Singular Value Decomposition."""

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.u_reduced: Optional[np.ndarray] = None
        self.s_reduced: Optional[np.ndarray] = None
        self.vt_reduced: Optional[np.ndarray] = None   # right singular vectors (components)
        self.mean: Optional[np.ndarray] = None        # for centering new data

    def fit(self, X: np.ndarray) -> None:
        """Fit SVD by computing full decomposition and storing components."""
        try:
            logger.info(f"Fitting SVD for {self.n_components} components.")
            # Center data (SVD on centered data gives same as PCA)
            self.mean = np.mean(X, axis=0)
            x_centered = X - self.mean

            U, S, Vt = np.linalg.svd(x_centered, full_matrices=False)
            self.total_global_variance = np.sum(S**2)

            self.u_reduced = U[:, :self.n_components]
            self.s_reduced = S[:self.n_components]
            self.vt_reduced = Vt[:self.n_components, :]   # principal components (like PCA)

            logger.success("SVD fitting successful.")
        except Exception as e:
            logger.error(f"SVD fitting failed: {e}")
            raise

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform using the left singular vectors scaled by singular values."""
        self.fit(X)
        # Project onto principal components (same as PCA transform)
        x_centered = X - self.mean
        return x_centered @ self.vt_reduced.T   # shape (n_samples, n_components)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project new data onto the learned principal components."""
        if self.vt_reduced is None or self.mean is None:
            raise ValueError("Model not fitted. Call fit() first.")
        x_centered = X - self.mean
        return x_centered @ self.vt_reduced.T

    def get_explained_variance_ratio(self) -> np.ndarray:
        if self.s_reduced is None:
            raise ValueError("Model must be fitted.")
        return (self.s_reduced**2) / self.total_global_variance