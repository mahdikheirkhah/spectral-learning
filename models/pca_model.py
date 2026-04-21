import numpy as np
from loguru import logger
from typing import Optional

class PCAModel:
    """Principal Component Analysis from scratch using Eigendecomposition."""

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components: Optional[np.ndarray] = None
        self.mean: Optional[np.ndarray] = None
        self.explained_variance: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> None:
        """Finds the principal components of the dataset."""
        try:
            logger.info(f"Fitting PCA for {self.n_components} components.")
            # 1. Center the data
            self.mean = np.mean(X, axis=0)
            x_centered = X - self.mean

            # 2. Covariance matrix
            cov_matrix = np.cov(x_centered, rowvar=False)

            # 3. Eigen-decomposition (eigh is optimized for symmetric matrices)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            # 4. Sort by highest eigenvalues
            idx = np.argsort(eigenvalues)[::-1]
            self.explained_variance = eigenvalues[idx][:self.n_components]
            self.components = eigenvectors[:, idx][:, :self.n_components]

            logger.success("PCA fitting successful.")
        except Exception as e:
            logger.error(f"PCA fitting failed: {e}")
            raise

    def get_explained_variance_ratio(self) -> np.ndarray:
        """
        Calculates the percentage of variance explained by each selected component.
        
        Returns:
            np.ndarray: Array of ratios summing toward 1.0.
        """
        if self.explained_variance is None:
            raise ValueError("Model must be fitted before calculating variance ratio.")
        
        # In PCA, the sum of all eigenvalues equals the total variance of the data
        total_variance = np.sum(self.explained_variance) 
        return self.explained_variance / total_variance
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Projects data into the principal component space."""
        if self.components is None:
            raise ValueError("Model not fitted.")
        return np.dot(X - self.mean, self.components)