import numpy as np
from loguru import logger
from typing import Optional
from utils.matrix_operations import compute_truncated_svd

class SVDModel:
    """Dimensionality reduction using Singular Value Decomposition."""

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.u_reduced: Optional[np.ndarray] = None
        self.s_reduced: Optional[np.ndarray] = None
        
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        try:
            # Perform SVD and get ALL singular values
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            
            # NEW: Store the sum of squares of ALL singular values
            self.total_global_variance = np.sum(S**2)
            
            # Keep only the top k
            self.u_reduced = U[:, :self.n_components]
            self.s_reduced = S[:self.n_components]
            
            return self.u_reduced * self.s_reduced
        except Exception as e:
            logger.error(f"SVD failed: {e}")
            raise

    def get_explained_variance_ratio(self) -> np.ndarray:
        if self.s_reduced is None:
            raise ValueError("Model must be fitted.")
        
        # Variance in SVD is the square of the singular values
        return (self.s_reduced**2) / self.total_global_variance