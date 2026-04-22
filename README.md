# Spectral Learning: PCA & SVD from Scratch

## Project Overview

This project implements **Principal Component Analysis (PCA)** and **Singular Value Decomposition (SVD)** from scratch to perform dimensionality reduction on high‑dimensional data. Using the **Wine Quality dataset** (red and white variants), the system reduces 11 physicochemical features to a lower‑dimensional space while preserving maximum variance. The implementations rely only on NumPy for core linear algebra (`linalg.eig`, `linalg.svd`); scikit‑learn is used **only** for clustering (K‑means) and evaluation metrics.

The goal is to understand the mathematical underpinnings of spectral methods and compare their effectiveness in terms of variance explained, clustering separability, and interpretability.

---

## Project Objectives

1. **Implement PCA from scratch**  
   - Mean center the data  
   - Compute the covariance matrix  
   - Perform eigendecomposition using `np.linalg.eigh`  
   - Sort eigenvalues/vectors and select top `k` components  
   - Project data onto the principal components  

2. **Implement SVD from scratch**  
   - Decompose the centered data matrix using `np.linalg.svd`  
   - Retain top `k` singular values and corresponding right singular vectors  
   - Project data using the right singular vectors (making SVD equivalent to PCA)  

3. **Evaluate dimensionality reduction**  
   - Compute variance explained per component  
   - Determine optimal `k` using a 95% cumulative variance threshold  
   - Apply K‑means clustering on reduced data and evaluate with silhouette score and Calinski‑Harabasz index  

4. **Visualize and interpret**  
   - Plot cumulative variance curves  
   - Show 2D/3D scatter plots of clusters (PCA and SVD)  
   - Display side‑by‑side variance comparison  
   - Visualise feature loadings (which original features contribute most to each principal component)  

5. **Compare PCA and SVD** – highlight theoretical and practical differences (see section below).

---

## Challenges Encountered & Solutions

| Challenge | Solution |
|-----------|----------|
| **Ensuring PCA and SVD give comparable reduced spaces** | Centered the data for both methods. In SVD, projected using **right singular vectors** \( V_k \) instead of \( U_k S_k \), aligning with PCA’s component space. |
| **Choosing the number of components objectively** | Implemented automatic selection based on 95% cumulative explained variance. This prevents over‑retaining noise components and avoids arbitrary choices. |
| **Interpreting principal components** | Added functions to print top‑3 original features per component and a heatmap of loadings, making the results interpretable for business stakeholders. |
| **Handling duplicate rows and missing values** | Used `drop_duplicates()` and `dropna()`; the Wine Quality dataset had ~15% duplicates in the white wine file, which were removed. |
| **Clustering evaluation** | Used silhouette score (internal validation) and Calinski‑Harabasz index to measure cluster separability in reduced space. Both scores were identical for PCA and SVD, confirming mathematical equivalence. |
| **Reducing dimensionality aggressively** | The 95% threshold retained 9 out of 11 features (97.8% variance). For a stronger reduction, the threshold can be lowered (e.g., 80% retains ~4–5 components). |

---

## Differences Between PCA and SVD

Although PCA and SVD are closely related, they differ in formulation, computation, and typical use cases.

| Aspect | Principal Component Analysis (PCA) | Singular Value Decomposition (SVD) |
|--------|-------------------------------------|-------------------------------------|
| **Mathematical formulation** | Eigendecomposition of the covariance matrix \( \frac{1}{n-1} X_c^T X_c \) | Decomposition of the data matrix \( X_c = U \Sigma V^T \) |
| **Input matrix** | Covariance matrix (symmetric, \( p \times p \)) | Any matrix (here, centered data \( n \times p \)) |
| **Output** | Eigenvalues (variance) and eigenvectors (principal components) | Singular values, left singular vectors \( U \), right singular vectors \( V \) |
| **Relation to variance** | Eigenvalues directly represent variance along each principal component | Squared singular values \( s_i^2 \) are proportional to variance (up to \( n-1 \)) |
| **Dimensionality reduction projection** | \( X_c W_k \) where \( W_k \) are top eigenvectors | \( X_c V_k \) (if using right singular vectors) |
| **Computational stability** | Slightly less stable for ill‑conditioned covariance matrices | More numerically stable, especially for near‑singular matrices |
| **Common applications** | Data visualization, noise reduction, feature extraction | Matrix approximation, recommendation systems, latent semantic analysis |
| **Interpretability** | Components are linear combinations of original features (loadings) | Same when using \( V_k \); \( U_k \) gives a different coordinate system |
| **In this project** | Implemented via `np.linalg.eigh` on covariance matrix | Implemented via `np.linalg.svd` on centered data; projection uses \( V_k \) |

**Key takeaway**: When the data is centered and projection uses right singular vectors, **PCA and SVD produce identical reduced representations** (up to sign flips). This equivalence is why the variance explained and clustering scores in our results are exactly the same.

---

## Project Structure

```
spectral-learning-project/
│
├── data/
│   ├── winequality-red.csv
│   └── winequality-white.csv
│
├── models/
│   ├── pca_model.py          # PCA from scratch
│   ├── svd_model.py          # SVD from scratch
│   └── __init__.py
│
├── utils/
│   ├── data_loader.py        # Load, clean, standardize
│   ├── matrix_operations.py  # Centering, covariance, top eigenvectors, SVD truncation
│   ├── clustering.py         # K‑means and evaluation metrics
│   ├── visualizer.py         # Plotting (2D/3D, cumulative variance, loadings heatmap)
│   └── __init__.py
│
├── main.py                   # Orchestrates the entire pipeline
├── README.md                 # This file
└── requirements.txt
```

---

## Setup & Installation (Poetry)

This project uses [Poetry](https://python-poetry.org/) for dependency management and packaging.

1. **Install Poetry** (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Install Project Dependencies**:
   Navigate to the project root and run:
   ```bash
   poetry install
   ```

3. **Run the Analysis Pipeline**:
   The `main.py` script automatically processes both Red and White wine datasets:
   ```bash
   poetry run python -m main
   ```

4. **Run Tests**:
   Verify the scratch implementations of PCA, SVD, and Matrix Utilities:
   ```bash
   poetry run pytest
   ```

---

## Usage

The script automatically:
- Loads and cleans each wine dataset (red, then white)
- Standardizes features (zero mean, unit variance)
- Determines optimal number of components for 95% variance explained
- Fits PCA and SVD with that number
- Projects the data and applies K‑means clustering (3 clusters: poor, average, excellent quality)
- Prints variance ratios, clustering scores, and top feature loadings
- Displays cumulative variance plots, 2D/3D cluster plots, and a side‑by‑side variance comparison

**Customisation**:
- Change the variance threshold in `determine_optimal_k(..., threshold=0.95)`
- Adjust number of clusters in `ClusteringPipeline(n_clusters=3)`
- Modify the quality binning function `rebin_quality`

---

## Additional Considerations: Non-Linear Methods

**Non‑linear exploration** – We applied t‑SNE to the original data and PCA‑reduced data. The t‑SNE visualisations revealed that wine quality clusters are non‑linearly separable, which explains why linear PCA could only achieve a silhouette score of ~0.19. This demonstrates the value of non‑linear methods for discovering complex patterns, even though they are not used as a primary reduction tool in this project.

---

## Results Summary (Example – Red Wine)

- **Optimal components**: 9 (explain 97.8% of total variance)  
- **Clustering scores** (on 9‑dim reduced space):  
  - Silhouette: 0.193  
  - Calinski‑Harabasz: 275.6  
- **Top feature loadings** (PC1): alcohol (0.51), volatile acidity (-0.40), sulphates (0.39)  
- **Variance ratios** (first 3 components): 28.3%, 17.3%, 14.1%

Both PCA and SVD gave identical variance ratios and clustering scores, confirming theoretical equivalence.

---

## Overfitting Prevention

- **Variance threshold** (95%) automatically discards low‑variance components that are likely noise.
- **Standardization** prevents features with larger scales from dominating the principal components.
- **No hyperparameter tuning** – the component count is determined solely from the training variance profile.
- **Internal clustering validation** (silhouette) helps detect if the reduced representation still separates the data well.

---

## References

- Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems*, 47(4), 547‑553.
- NumPy documentation: [numpy.linalg.eig](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html), [numpy.linalg.svd](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html)
- Relationship between PCA and SVD: [StackExchange](https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca)

---

## License

This project is for educational purposes as part of a spectral learning assignment.

