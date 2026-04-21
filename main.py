from utils.data_loader import DataLoader
from loguru import logger


def main():
    # Update this path to your actual filename in the data/ folder
    path = "data/winequality-red.csv"

    loader = DataLoader(path)

    try:
        loader.load_data()
        loader.clean_data()
        X, y = loader.split_features_target(target_column="quality")

        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target vector shape: {y.shape}")
        logger.success("Phase 2 logic verified!")

    except Exception as e:
        logger.critical(f"Phase 2 verification failed: {e}")


if __name__ == "__main__":
    main()
