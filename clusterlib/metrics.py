#Copyright (c) 2026 Evgeniia VOLCHOK
#for contacts e.p.volchok@gmail.com

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors



def space_preservation(X, emb, k: int = 15) -> float:
    """Mean fraction of k-nearest neighbours preserved after embedding.

    Measures how well local neighbourhood structure in the original space X
    is retained in the embedded space emb.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    emb : array-like of shape (n_samples, n_components)
    k : int

    Returns
    -------
    score : float in [0, 1]
    """
    X = np.asarray(X); emb = np.asarray(emb)
    nn_orig = NearestNeighbors(n_neighbors=k).fit(X)
    nn_emb  = NearestNeighbors(n_neighbors=k).fit(emb)
    idx_orig = nn_orig.kneighbors(X,   return_distance=False)
    idx_emb  = nn_emb.kneighbors(emb,  return_distance=False)
    scores = [len(set(idx_orig[i]) & set(idx_emb[i])) / k
              for i in range(len(X))]
    return float(np.mean(scores))



def mst_separation(mst_df, cluster_labels) -> dict:
    """Per-cluster MST separation: max internal edge / min external edge.

    For each cluster returns the worst-case ratio across all neighbouring
    clusters.  A ratio > 1 means the cluster bleeds into its neighbours.

    Requires HDBSCAN fitted with gen_min_span_tree=True:
    mst_df = clusterer.minimum_spanning_tree_.to_pandas()

    Returns
    -------
    dict mapping cluster_id -> ratio (or None if no internal / external edges)
    """
    unique_clusters = sorted(set(cluster_labels) - {-1})
    result = {}

    for cid in unique_clusters:
        internal_mask = ((mst_df['label_from'] == cid) &
                         (mst_df['label_to']   == cid))
        internal = mst_df[internal_mask]['distance']

        if internal.empty:
            result[cid] = None
            continue

        external_mask = (
            ((mst_df['label_from'] == cid) & (mst_df['label_to'] != cid) &
             (mst_df['label_to'] != -1)) |
            ((mst_df['label_to'] == cid) & (mst_df['label_from'] != cid) &
             (mst_df['label_from'] != -1))
        )
        external_edges = mst_df[external_mask].copy()

        if external_edges.empty:
            result[cid] = None
            continue

        external_edges['neighbour'] = external_edges.apply(
            lambda r: r['label_to'] if r['label_from'] == cid else r['label_from'], axis=1
        )
        ratios = {nb: internal.max() / grp['distance'].min()
                  for nb, grp in external_edges.groupby('neighbour')}
        result[cid] = min(ratios.values())

    return result


def mst_cohesion(mst_df, cluster_labels) -> dict:
    """Per-cluster MST cohesion: internal edge statistics.

    Returns
    -------
    dict mapping cluster_id -> dict with keys
    max_edge, mean_edge, diameter (sum), n_edges
    (or None for single-point clusters).
    """
    df = mst_df.copy()
    if 'label_from' not in df.columns:
        df['label_from'] = cluster_labels[df['from'].astype(int).values]
        df['label_to']   = cluster_labels[df['to'].astype(int).values]

    result = {}
    for cid in set(cluster_labels) - {-1}:
        mask = (df['label_from'] == cid) & (df['label_to'] == cid)
        internal = df[mask]['distance']
        if internal.empty:
            result[cid] = None
            continue
        result[cid] = {
            'max_edge':  float(internal.max()),
            'mean_edge': float(internal.mean()),
            'diameter':  float(internal.sum()),
            'n_edges':   len(internal),
        }

    return result




def significant_components(h0_diagram, threshold=None) -> int:
    """Count H0 components with persistence above *threshold*.

    One component always has death=infty (the global connected component); it is
    excluded from the count so the function returns only finite significant
    components.

    If threshold is None it is set to median + 2 x std of finite
    persistence values.
    """
    finite = h0_diagram[h0_diagram[:, 1] != np.inf]
    persistence = finite[:, 1] - finite[:, 0]
    if len(persistence) == 0:
        return 0
    if threshold is None:
        threshold = np.median(persistence) + 2 * persistence.std()
    return int(np.sum(persistence > threshold))


def persistence_entropy(diagram) -> float:
    """Shannon entropy of the H0 persistence distribution.

    Higher entropy means more evenly distributed persistence values (less
    clear cluster structure).  Low entropy with one dominant component
    suggests a single tight cluster.
    """
    finite = diagram[diagram[:, 1] != np.inf]
    persistence = finite[:, 1] - finite[:, 0]
    total = persistence.sum()
    if total == 0:
        return 0.0
    p = persistence / total
    return float(-np.sum(p * np.log(p + 1e-10)))


def significant_components_by_scale(h0_diagram, n_top: int = 10) -> int:
    """Identify the elbow in sorted persistence values to count significant structures.

    Plots the top-n_top persistence values and marks the detected elbow.
    Returns the number of structures before the steepest drop.
    """
    finite = h0_diagram[h0_diagram[:, 1] != np.inf]
    persistence = finite[:, 1] - finite[:, 0]
    sorted_p = np.sort(persistence)[::-1]

    diffs = np.diff(sorted_p)
    elbow = int(np.argmin(diffs[:n_top])) + 1

    xs = list(range(1, min(n_top, len(sorted_p)) + 1))
    per_val = sorted_p[:n_top].tolist()
    plt.plot(xs, per_val)
    plt.xlabel('Number of structures')
    plt.ylabel('Persistence')

    return elbow + 1
