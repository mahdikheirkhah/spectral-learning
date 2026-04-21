import pytest
import numpy as np
from models.svd_model import SVDModel

@pytest.fixture
def sample_matrix():
    """Creates a 5x3 matrix for testing SVD."""
    return np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0]
    ])

## --- Happy Path Tests ---

def test_svd_fit_transform_shape(sample_matrix):
    """Verifies that the output projection has the correct dimensions."""
    n_comp = 2
    model = SVDModel(n_components=n_comp)
    projection = model.fit_transform(sample_matrix)
    
    # Rows should match input rows (5), columns should match n_components (2)
    assert projection.shape == (5, n_comp)
    # Verify internal attributes are populated
    assert model.u_reduced.shape == (5, n_comp)
    assert model.s_reduced.shape == (n_comp,)

def test_svd_reproducibility(sample_matrix):
    """Ensures that running SVD on the same data twice yields the same result."""
    model = SVDModel(n_components=1)
    res1 = model.fit_transform(sample_matrix)
    res2 = model.fit_transform(sample_matrix)
    
    assert np.allclose(res1, res2)

## --- Edge Cases & Exception Handling ---

def test_svd_invalid_n_components(sample_matrix):
    """Should raise ValueError if n_components exceeds matrix rank."""
    # Matrix is 5x3, so max components is 3.
    model = SVDModel(n_components=10)
    with pytest.raises(ValueError):
        model.fit_transform(sample_matrix)

def test_svd_empty_input():
    """Should raise an error when an empty array is passed."""
    model = SVDModel(n_components=1)
    with pytest.raises(Exception):
        model.fit_transform(np.array([[]]))

def test_svd_invalid_data_type():
    """Ensures the model propagates errors for non-numeric data."""
    model = SVDModel(n_components=1)
    with pytest.raises(Exception):
        model.fit_transform(np.array([["a", "b"], ["c", "d"]]))

def test_svd_unfitted_attributes():
    """Verifies that attributes are None before fitting."""
    model = SVDModel(n_components=2)
    assert model.u_reduced is None
    assert model.s_reduced is None