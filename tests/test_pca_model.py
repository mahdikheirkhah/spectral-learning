import pytest
import numpy as np
from models.pca_model import PCAModel

@pytest.fixture
def synthetic_data():
    """
    Creates a simple 2D dataset where the first feature has much 
    higher variance than the second.
    """
    np.random.seed(42)
    x1 = np.random.normal(0, 10, 100) # High variance
    x2 = np.random.normal(0, 1, 100)  # Low variance
    return np.column_stack((x1, x2))

## --- Happy Path Tests ---

def test_pca_fit_success(synthetic_data):
    """Verifies that fit populates the model attributes correctly."""
    n_comp = 1
    model = PCAModel(n_components=n_comp)
    model.fit(synthetic_data)
    
    assert model.components is not None
    assert model.mean is not None
    assert model.explained_variance is not None
    assert model.components.shape == (2, n_comp)
    assert len(model.explained_variance) == n_comp

def test_pca_transform_shape(synthetic_data):
    """Verifies that transformation reduces dimensions correctly."""
    n_comp = 1
    model = PCAModel(n_components=n_comp)
    model.fit(synthetic_data)
    X_transformed = model.transform(synthetic_data)
    
    assert X_transformed.shape == (100, n_comp)

def test_explained_variance_ratio(synthetic_data):
    """
    In our synthetic data, the first component should capture 
    nearly 100% of the variance.
    """
    model = PCAModel(n_components=2)
    model.fit(synthetic_data)
    ratios = model.get_explained_variance_ratio()
    
    assert len(ratios) == 2
    assert ratios[0] > ratios[1]
    assert np.isclose(np.sum(ratios), 1.0)

## --- Edge Cases & Exception Handling ---

def test_transform_before_fit(synthetic_data):
    """Should raise ValueError if transform is called without fitting."""
    model = PCAModel(n_components=1)
    with pytest.raises(ValueError, match="Model not fitted"):
        model.transform(synthetic_data)

def test_get_ratio_before_fit():
    """Should raise ValueError if ratio is requested without fitting."""
    model = PCAModel(n_components=1)
    with pytest.raises(ValueError, match="Model must be fitted"):
        model.get_explained_variance_ratio()

def test_pca_fit_failure_invalid_data():
    """Tests how the model handles invalid input via the utility errors."""
    model = PCAModel(n_components=1)
    with pytest.raises(Exception): # Should propagate the error from center_data
        model.fit(np.array(["string", "data"]))

def test_pca_consistency(synthetic_data):
    """
    Tests that fitting twice with the same data yields the same result 
    (Idempotency).
    """
    model = PCAModel(n_components=1)
    model.fit(synthetic_data)
    comp1 = model.components.copy()
    
    model.fit(synthetic_data)
    comp2 = model.components.copy()
    
    assert np.allclose(comp1, comp2)