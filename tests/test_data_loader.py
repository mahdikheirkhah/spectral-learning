import pytest
import pandas as pd
import numpy as np
from utils.data_loader import DataLoader

@pytest.fixture
def valid_csv(tmp_path):
    """Creates a valid temporary CSV file for testing."""
    d = tmp_path / "subdir"
    d.mkdir()
    file = d / "test_wine.csv"
    # Create sample wine data (standardized-ready)
    df = pd.DataFrame({
        "fixed_acidity": [7.4, 7.8, 7.8, 11.2],
        "volatile_acidity": [0.7, 0.88, 0.76, 0.28],
        "quality": [5, 5, 5, 6]
    })
    df.to_csv(file, index=False)
    return str(file)

@pytest.fixture
def empty_csv(tmp_path):
    """Creates an empty file."""
    file = tmp_path / "empty.csv"
    file.write_text("")
    return str(file)

## --- Happy Path Tests ---

def test_load_data_success(valid_csv):
    loader = DataLoader(valid_csv)
    df = loader.load_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4

def test_full_pipeline_flow(valid_csv):
    """Tests the entire sequential flow: Load -> Clean -> Split -> Standardize."""
    loader = DataLoader(valid_csv)
    loader.load_data()
    loader.clean_data()
    X, y = loader.split_features_target(target_column="quality")
    X_scaled = loader.standardize_features()
    
    assert X_scaled.shape == (4, 2)
    assert len(y) == 4
    # Check if standardization worked (Mean should be approx 0)
    assert np.allclose(X_scaled.mean(axis=0), [0, 0], atol=1e-7)

## --- Edge Case & Exception Tests ---

def test_file_not_found():
    loader = DataLoader("non_existent_file.csv")
    with pytest.raises(FileNotFoundError):
        loader.load_data()

def test_clean_data_handles_duplicates(tmp_path):
    file = tmp_path / "dup.csv"
    df = pd.DataFrame({"a": [1, 1], "b": [2, 2], "quality": [5, 5]})
    df.to_csv(file, index=False)
    
    loader = DataLoader(str(file))
    loader.load_data()
    cleaned_df = loader.clean_data()
    # Should drop one row
    assert len(cleaned_df) == 1

def test_clean_data_handles_nans(tmp_path):
    file = tmp_path / "nan.csv"
    df = pd.DataFrame({"a": [1, np.nan], "quality": [5, 6]})
    df.to_csv(file, index=False)
    
    loader = DataLoader(str(file))
    loader.load_data()
    cleaned_df = loader.clean_data()
    assert len(cleaned_df) == 1

def test_split_features_invalid_column(valid_csv):
    loader = DataLoader(valid_csv)
    loader.load_data()
    # 'wrong_column' does not exist
    with pytest.raises(KeyError):
        loader.split_features_target(target_column="wrong_column")

def test_standardize_before_split(valid_csv):
    """Test state error: trying to standardize without having features (X)."""
    loader = DataLoader(valid_csv)
    loader.load_data()
    with pytest.raises(ValueError, match="Feature matrix \(X\) is empty"):
        loader.standardize_features()

def test_standardize_non_numeric_data(tmp_path):
    """Spectral methods fail on strings. DataLoader should handle/propagate that error."""
    file = tmp_path / "text.csv"
    df = pd.DataFrame({"a": ["high", "low"], "quality": [5, 6]})
    df.to_csv(file, index=False)
    
    loader = DataLoader(str(file))
    loader.load_data()
    loader.split_features_target("quality")
    with pytest.raises(Exception): # StandardScaler will raise a TypeError/ValueError
        loader.standardize_features()