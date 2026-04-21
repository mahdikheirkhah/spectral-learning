import numpy as np
from loguru import logger
from typing import Optional

class SVDModel:
    """Dimensionality reduction using Singular Value Decomposition."""

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.u_reduced: Optional[np.ndarray] = None
        self.s_reduced: Optional[np.ndarray] = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Decomposes X into U, S, V^T and returns the reduced projection."""
        try:
            logger.info(f"Performing SVD for {self.n_components} components.")
            
            # Full SVD: X = U * S * Vt
            U, S, Vt = np.linalg.svd(X, full_matrices=False)

            # Keep top k components
            self.u_reduced = U[:, :self.n_components]
            self.s_reduced = S[:self.n_components]
            
            # The projected data is U * S
            projection = self.u_reduced * self.s_reduced
            
            logger.success("SVD decomposition complete.")
            return projection
        except Exception as e:
            logger.error(f"SVD failed: {e}")
            raise
        