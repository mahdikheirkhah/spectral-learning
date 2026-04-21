from utils.data_loader import DataLoader
from models.pca_model import PCAModel
from models.svd_model import SVDModel
from loguru import logger

def main():
    try:
        # 1. Pipeline: Load & Preprocess
        loader = DataLoader("data/winequality-red.csv")
        loader.load_data()
        loader.clean_data()
        X, y = loader.split_features_target(target_column="quality")
        X_scaled = loader.standardize_features()

        # 2. Spectral Method: PCA
        pca = PCAModel(n_components=2)
        pca.fit(X_scaled)
        X_pca = pca.transform(X_scaled)
        
        # Calculate Information Retention
        ratios = pca.get_explained_variance_ratio()
        logger.info(f"PCA Variance Ratios: {ratios}")
        logger.success(f"Total Variance Captured: {sum(ratios)*100:.2f}%")

        # 3. Spectral Method: SVD
        svd = SVDModel(n_components=2)
        X_svd = svd.fit_transform(X_scaled)
        
        logger.success("Spectral Decomposition Pipeline Complete.")

    except Exception as e:
        logger.critical(f"Pipeline crashed: {e}")

if __name__ == "__main__":
    main()