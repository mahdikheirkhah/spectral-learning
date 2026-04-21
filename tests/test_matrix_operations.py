import pytest
import numpy as np
from utils.matrix_operations import (
    center_data,
    compute_covariance_matrix,
    compute_top_eigenvectors,
    compute_truncated_svd,
)

@pytest.fixture
def sample_data():
    """Provides a simple 2D array for testing."""
    return np.array([
        [2.5, 2.4],
        [0.5, 0.7],
        [2.2, 2.9],
        [1.9, 2.2],
        [3.1, 3.0],
        [2.3, 2.7],
        [2.0, 1.6],
        [1.0, 1.1],
        [1.5, 1.6],
        [1.1, 0.9]
    ])

## --- Tests for center_data ---

def test_center_data_success(sample_data):
    X_centered, mean = center_data(sample_data)
    
    # Check shape
    assert X_centered.shape == sample_data.shape
    # Check if mean of centered data is practically 0
    assert np.allclose(np.mean(X_centered, axis=0), [0.0, 0.0], atol=1e-7)
    # Check extracted mean
    assert np.allclose(mean, np.mean(sample_data, axis=0))

def test_center_data_empty():
    with pytest.raises(ValueError, match="cannot be empty"):
        center_data(np.array([]))

def test_center_data_invalid_type():
    with pytest.raises(TypeError, match="must be a NumPy array"):
        center_data([1, 2, 3]) # Passing a list instead of ndarray

## --- Tests for compute_covariance_matrix ---

def test_compute_covariance_success(sample_data):
    X_centered, _ = center_data(sample_data)
    cov_matrix = compute_covariance_matrix(X_centered)
    
    # Covariance matrix for N features should be N x N
    assert cov_matrix.shape == (2, 2)
    # Covariance matrix must be symmetric
    assert np.allclose(cov_matrix, cov_matrix.T)

def test_compute_covariance_invalid_shape():
    with pytest.raises(ValueError, match="must be a 2D NumPy array"):
        compute_covariance_matrix(np.array([1, 2, 3])) # 1D array

## --- Tests for compute_top_eigenvectors ---

def test_compute_top_eigenvectors_success(sample_data):
    X_centered, _ = center_data(sample_data)
    cov_matrix = compute_covariance_matrix(X_centered)
    
    n_components = 1
    eigenvals, eigenvecs = compute_top_eigenvectors(cov_matrix, n_components)
    
    assert eigenvals.shape == (1,)
    assert eigenvecs.shape == (2, 1) # 2 features, 1 component

def test_compute_top_eigenvectors_not_square():
    non_square_matrix = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError, match="must be square"):
        compute_top_eigenvectors(non_square_matrix, 1)

def test_compute_top_eigenvectors_invalid_components(sample_data):
    X_centered, _ = center_data(sample_data)
    cov_matrix = compute_covariance_matrix(X_centered)
    
    with pytest.raises(ValueError, match="Invalid n_components"):
        compute_top_eigenvectors(cov_matrix, 5) # Only 2 features available
        
    with pytest.raises(ValueError, match="Invalid n_components"):
        compute_top_eigenvectors(cov_matrix, 0) # Cannot be 0

## --- Tests for compute_truncated_svd ---

def test_compute_truncated_svd_success(sample_data):
    n_components = 1
    u, s, vt = compute_truncated_svd(sample_data, n_components)
    
    # Check truncated shapes
    assert u.shape == (10, 1) # 10 samples, 1 component
    assert s.shape == (1,)    # 1 singular value
    assert vt.shape == (1, 2) # 1 component, 2 features

def test_compute_truncated_svd_invalid_components(sample_data):
    # Max components is min(10, 2) = 2
    with pytest.raises(ValueError, match="Invalid n_components"):
        compute_truncated_svd(sample_data, 3)

def test_compute_truncated_svd_invalid_shape():
    with pytest.raises(ValueError, match="must be a 2D NumPy array"):
        compute_truncated_svd(np.array([1, 2, 3]), 1)