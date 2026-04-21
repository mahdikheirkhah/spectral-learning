import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional


class DataLoader:
    """
    A generalized data loader and preprocessor.
    Provides methods for loading, cleaning (dropping NAs/duplicates),
    splitting features/targets, and standardization.
    """

    def __init__(self, file_path: str):
        """
        Initializes the loader with the path to the dataset.

        Args:
            file_path (str): Path to the dataset (CSV).
        """
        self.file_path = Path(file_path)
        self.data: Optional[pd.DataFrame] = None
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None

    def load_data(self) -> pd.DataFrame:
        """
        Loads the CSV data from the specified path.
        Uses automated delimiter detection to handle different CSV formats.

        Returns:
            pd.DataFrame: The raw loaded data.
        """
        try:
            logger.info(f"Loading data from {self.file_path}")
            if not self.file_path.exists():
                raise FileNotFoundError(f"File not found: {self.file_path}")

            self.data = pd.read_csv(self.file_path, sep=None, engine="python")
            logger.success(f"Successfully loaded {len(self.data)} instances.")
            return self.data

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def clean_data(self) -> pd.DataFrame:
        """
        Cleans the dataset by removing duplicated rows and dropping any 
        rows containing missing values (NaNs).

        Returns:
            pd.DataFrame: The cleaned dataframe.
        """
        try:
            if self.data is None:
                self.load_data()

            initial_len = len(self.data)
            
            # Drop duplicates and NAs
            self.data = self.data.drop_duplicates().dropna()
            
            final_len = len(self.data)
            logger.info(f"Cleaning complete: Dropped {initial_len - final_len} rows (duplicates/NAs).")
            return self.data

        except Exception as e:
            logger.error(f"Error during data cleaning: {e}")
            raise

    def split_features_target(self, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Separates the dataframe into a feature matrix (X) and a target vector (y).

        Args:
            target_column (str): The name of the column to use as the target.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (Features, Target)
        """
        try:
            if self.data is None:
                self.clean_data()

            logger.info(f"Splitting data into features and target: '{target_column}'")
            
            X_df = self.data.drop(columns=[target_column])
            self.y = self.data[target_column].values
            self.X = X_df.values

            return self.X, self.y

        except Exception as e:
            logger.error(f"Error during feature splitting: {e}")
            raise

    def standardize_features(self) -> np.ndarray:
        """
        Applies Z-score standardization to the feature matrix.
        Ensures each feature has Mean=0 and Unit Variance (Std=1).
        This is a critical step for Spectral methods like PCA and SVD.

        Returns:
            np.ndarray: The standardized feature matrix.
        """
        try:
            if self.X is None:
                raise ValueError("Feature matrix (X) is empty. Run split_features_target first.")

            logger.info("Applying standardization to features.")
            scaler = StandardScaler()
            self.X = scaler.fit_transform(self.X)
            
            logger.success("Standardization complete.")
            return self.X

        except Exception as e:
            logger.error(f"Standardization failed: {e}")
            raise