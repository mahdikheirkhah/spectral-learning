import numpy as np
from typing import Tuple
from loguru import logger

def center_data(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Centers the data by subtracting the mean of each column.
    
    Args:
        X (np.ndarray): The input matrix.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: The centered matrix and the computed mean vector.
    """
    try:
        if not isinstance(X, np.ndarray):
            raise TypeError("Input must be a NumPy array.")
        if X.size == 0:
            raise ValueError("Input array cannot be empty.")
            
        mean = np.mean(X, axis=0)
        X_centered = X - mean
        return X_centered, mean
    except Exception as e:
        logger.error(f"Error in center_data: {e}")
        raise

def compute_covariance_matrix(X_centered: np.ndarray) -> np.ndarray:
    """
    Computes the covariance matrix of centered data.
    
    Args:
        X_centered (np.ndarray): The centered input matrix.
        
    Returns:
        np.ndarray: The covariance matrix.
    """
    try:
        if not isinstance(X_centered, np.ndarray) or X_centered.ndim != 2:
            raise ValueError("Input must be a 2D NumPy array.")
            
        return np.cov(X_centered, rowvar=False)
    except Exception as e:
        logger.error(f"Error in compute_covariance_matrix: {e}")
        raise

def compute_top_eigenvectors(
    cov_matrix: np.ndarray, n_components: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the eigendecomposition of a covariance matrix and returns 
    the top k eigenvalues and eigenvectors.
    
    Args:
        cov_matrix (np.ndarray): The covariance matrix.
        n_components (int): Number of top components to return.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Top eigenvalues and top eigenvectors.
    """
    try:
        if not isinstance(cov_matrix, np.ndarray) or cov_matrix.ndim != 2:
            raise ValueError("Covariance matrix must be a 2D array.")
        if cov_matrix.shape[0] != cov_matrix.shape[1]:
            raise ValueError("Covariance matrix must be square.")
        if n_components <= 0 or n_components > cov_matrix.shape[1]:
            raise ValueError(f"Invalid n_components: {n_components}. Must be between 1 and {cov_matrix.shape[1]}.")

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by descending eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        
        top_eigenvalues = eigenvalues[idx][:n_components]
        top_eigenvectors = eigenvectors[:, idx][:, :n_components]
        
        return top_eigenvalues, top_eigenvectors
    except Exception as e:
        logger.error(f"Error in compute_top_eigenvectors: {e}")
        raise

def compute_truncated_svd(
    X: np.ndarray, n_components: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs Singular Value Decomposition and returns truncated matrices.
    
    Args:
        X (np.ndarray): The input matrix.
        n_components (int): Number of top components to retain.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Truncated U, S, and V^T matrices.
    """
    try:
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ValueError("Input must be a 2D NumPy array.")
        
        max_components = min(X.shape)
        if n_components <= 0 or n_components > max_components:
            raise ValueError(f"Invalid n_components: {n_components}. Must be between 1 and {max_components}.")

        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        
        u_reduced = U[:, :n_components]
        s_reduced = S[:n_components]
        vt_reduced = Vt[:n_components, :]
        
        return u_reduced, s_reduced, vt_reduced
    except Exception as e:
        logger.error(f"Error in compute_truncated_svd: {e}")
        raise