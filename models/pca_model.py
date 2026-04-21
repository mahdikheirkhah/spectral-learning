import numpy as np
from loguru import logger
from typing import Optional
from utils.matrix_operations import (
    center_data,
    compute_covariance_matrix,
    compute_top_eigenvectors,
)

class PCAModel:
    """Principal Component Analysis from scratch using Eigendecomposition."""

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components: Optional[np.ndarray] = None
        self.mean: Optional[np.ndarray] = None
        self.explained_variance: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> None:
        try:
            logger.info(f"Fitting PCA for {self.n_components} components.")
            x_centered, self.mean = center_data(X)
            cov_matrix = compute_covariance_matrix(x_centered)
            
            # Get ALL eigenvalues first
            all_eigenvalues, all_eigenvectors = np.linalg.eigh(cov_matrix)
            
            # Sort them
            idx = np.argsort(all_eigenvalues)[::-1]
            sorted_eigenvalues = all_eigenvalues[idx]
            sorted_eigenvectors = all_eigenvectors[:, idx]

            # NEW: Store the total sum of ALL eigenvalues (Total Global Variance)
            self.total_global_variance = np.sum(sorted_eigenvalues)

            # Store only the top k components
            self.explained_variance = sorted_eigenvalues[:self.n_components]
            self.components = sorted_eigenvectors[:, :self.n_components]

            logger.success("PCA fitting successful.")
        except Exception as e:
            logger.error(f"PCA fitting failed: {e}")
            raise

    def get_explained_variance_ratio(self) -> np.ndarray:
        if self.explained_variance is None:
            raise ValueError("Model must be fitted.")
        
        # Divide the kept variance by the GLOBAL variance
        return self.explained_variance / self.total_global_variance
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Projects data into the principal component space."""
        if self.components is None or self.mean is None:
            raise ValueError("Model not fitted.")
        return np.dot(X - self.mean, self.components)