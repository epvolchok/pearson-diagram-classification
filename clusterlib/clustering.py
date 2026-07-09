#Copyright (c) 2026 Evgeniia VOLCHOK
#for contacts e.p.volchok@gmail.com

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import pandas as pd
import umap
import hdbscan


def run_pipeline(
    df,
    umap_params: dict,
    hdbscan_params: dict,
    random_state: int = 42,
    gen_min_span_tree: bool = False,
) -> np.ndarray:
    """Run UMAP dimensionality reduction followed by HDBSCAN clustering.

    Parameters
    ----------
    df : array-like or DataFrame of shape (n_samples, n_features)
    umap_params : dict
        Keyword arguments forwarded to umap.UMAP (excluding random_state).
    hdbscan_params : dict
        Keyword arguments forwarded to hdbscan.HDBSCAN.
    random_state : int
        Random seed for UMAP.
    gen_min_span_tree : bool
        Whether to generate the minimum spanning tree in HDBSCAN.  Needed for
        MST-based diagnostics; set to False to save memory in grid searches.

    Returns
    -------
    labels : np.ndarray of shape (n_samples,)
        Cluster labels; -1 denotes noise.
    """
    X = df.values if hasattr(df, 'values') else np.asarray(df)
    reducer = umap.UMAP(random_state=random_state, **umap_params)
    X_umap = reducer.fit_transform(X)
    clusterer = hdbscan.HDBSCAN(gen_min_span_tree=gen_min_span_tree, **hdbscan_params)
    return clusterer.fit_predict(X_umap)


def embed_2d(df, umap_params: dict, random_state: int = 42) -> np.ndarray:
    """Create a 2-D UMAP embedding for visualisation.

    Overrides n_components=2 in *umap_params* so the function always
    returns a 2-D array regardless of the high-dimensional UMAP config.

    Returns
    -------
    X_2d : np.ndarray of shape (n_samples, 2)
    """
    X = df.values if hasattr(df, 'values') else np.asarray(df)
    params = {**umap_params, 'n_components': 2}
    return umap.UMAP(random_state=random_state, **params).fit_transform(X)


def n_clusters(labels) -> int:
    """Count non-noise clusters (i.e. labels != -1)."""
    labels = np.asarray(labels)
    return len(set(labels)) - (1 if -1 in labels else 0)


def get_labels(
    X,
    umap_grid: list[dict],
    hdbscan_grid: list[dict],
    seeds=(123,),
) -> list[np.ndarray]:
    """Generate cluster labels for every combination of UMAP / HDBSCAN params and seeds.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    umap_grid : list of dicts, each passed to umap.UMAP
    hdbscan_grid : list of dicts, each passed to hdbscan.HDBSCAN
    seeds : sequence of ints
        Random seeds for UMAP.

    Returns
    -------
    all_labels : list of np.ndarray, one per (umap_params, seed, hdbscan_params) combination
    """
    all_labels = []
    X = X.values if hasattr(X, 'values') else np.asarray(X)
    for up in umap_grid:
        for seed in seeds:
            reducer = umap.UMAP(random_state=seed, **up)
            X_umap = reducer.fit_transform(X)
            for hp in hdbscan_grid:
                labels = hdbscan.HDBSCAN(**hp).fit_predict(X_umap)
                all_labels.append(labels)
    return all_labels


def find_epsilon(clusterer) -> None:
    """Print cluster merge events from the HDBSCAN condensed tree.

    Useful for visually identifying a suitable cluster_selection_epsilon.
    Requires HDBSCAN fitted with gen_min_span_tree=True.
    """
    tree = clusterer.condensed_tree_.to_pandas()
    cluster_merges = (tree[tree['child_size'] > 1]
                      .sort_values('lambda_val', ascending=False)
                      .reset_index(drop=True))
    print(cluster_merges[['child', 'child_size', 'lambda_val']].head(30))

    for _, row in cluster_merges.iterrows():
        parent = row['parent']
        parent_size = tree[tree['child'] == parent]['child_size'].values
        if len(parent_size) > 0 and row['child_size'] < 20:
            print(f"Cluster size {int(row['child_size'])} absorbed at "
                  f"lambda={row['lambda_val']:.3f} "
                  f"(epsilon ~ {1/row['lambda_val']:.3f})")


def find_epsilon_lambda(
    clusterer,
    lambda_target: float = 1.4,
    delta: float = 0.2,
) -> None:
    """Print cluster merge events within a lambda window.

    Narrows the output of :func:`find_epsilon` to a specific lambda range so
    you can focus on candidate epsilon values near a known merge point.
    """
    tree = clusterer.condensed_tree_.to_pandas()
    cluster_merges = (tree[tree['child_size'] > 1]
                      .sort_values('lambda_val', ascending=False)
                      .reset_index(drop=True))
    window = cluster_merges[
        (cluster_merges['lambda_val'] >= lambda_target - delta) &
        (cluster_merges['lambda_val'] <= lambda_target + delta)
    ].copy()

    for _, row in window.iterrows():
        parent = row['parent']
        parent_children = tree[tree['parent'] == parent]
        parent_size = parent_children['child_size'].sum()
        print(f"child={int(row['child']):6d} | size={int(row['child_size']):4d} | "
              f"lambda={row['lambda_val']:.3f} | epsilon~{1/row['lambda_val']:.3f} | "
              f"parent_size={int(parent_size):4d} | "
              f"size_ratio={row['child_size']/parent_size:.2f}")
