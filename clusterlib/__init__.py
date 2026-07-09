"""
Modules
-------
preprocessing   Feature selection: variance filter, correlation removal
clustering      UMAP+HDBSCAN pipeline, parameter utilities, HDBSCAN diagnostics
stability       Stability metrics: ARI, neighbourhood preservation, bootstrap, seed
metrics         Embedding quality: space preservation, MST metrics, TDA
visualization   Plotting: stability contour maps, ARI boxplots, cluster scatter
pareto          Pareto front computation and CSV/Parquet I/O
gradcam         Grad-CAM visualisation for cluster medoids (requires torch + cv2)
io              Save/load .npz results for stability, bootstrap, grid search

"""
