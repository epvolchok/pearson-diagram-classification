#Copyright (c) 2026 Evgeniia VOLCHOK
#for contacts e.p.volchok@gmail.com

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0


import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score

from .clustering import run_pipeline, get_labels


def ari_core_weighted(a, b) -> tuple[float, float]:
    """Adjusted Rand Index restricted to points that are non-noise in *both* runs.

    Returns
    -------
    ari : float
    coverage : float
        Fraction of points that are non-noise in both a and b.
    """
    a = np.asarray(a); b = np.asarray(b)
    mask = (a >= 0) & (b >= 0)
    coverage = float(mask.mean())
    if mask.sum() < 2:
        return 0.0, coverage
    ari = adjusted_rand_score(a[mask], b[mask])
    return float(ari), coverage


def label_preservation(labels_a, labels_b) -> np.ndarray:
    """Per-point preservation using cluster co-membership as neighbourhood.

    For each point i the neighbourhood is defined as all points that share
    the same cluster label in labels_a.  The score is the fraction of those
    neighbours that also share i's cluster in labels_b.

    Noise points (label == -1) in labels_a receive a score of 0.

    Returns
    -------
    stability : np.ndarray of shape (n_samples,)
    """
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    n = len(labels_a)
    stability = np.zeros(n)

    for i in range(n):
        if labels_a[i] < 0:
            continue
        neighbours = np.where(labels_a == labels_a[i])[0]
        neighbours = neighbours[neighbours != i]
        if len(neighbours) == 0:
            continue
        if labels_b[i] < 0:
            stability[i] = 0.0
        else:
            stability[i] = np.sum(labels_b[neighbours] == labels_b[i]) / len(neighbours)

    return stability


def label_preservation_knn(labels_a, labels_b, X, k: int = 15) -> np.ndarray:
    """Per-point preservation using kNN in feature space as neighbourhood.

    For each point i the neighbourhood is the k nearest neighbours in X.
    The score is the fraction of those neighbours for which both the cluster
    assignment in labels_a and in labels_b agree with i.

    Noise points in labels_a receive a score of 0.

    Returns
    -------
    stability : np.ndarray of shape (n_samples,)
    """
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    n = len(labels_a)
    stability = np.zeros(n)

    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    _, neighbor_idx = nbrs.kneighbors(X)

    for i in range(n):
        if labels_a[i] < 0:
            continue
        neighbours = neighbor_idx[i]
        if len(neighbours) == 0:
            continue
        if labels_b[i] < 0:
            stability[i] = 0.0
        else:
            same_in_a = labels_a[neighbours] == labels_a[i]
            same_in_b = labels_b[neighbours] == labels_b[i]
            denom = same_in_a.sum()
            stability[i] = np.sum(same_in_a & same_in_b) / denom if denom > 0 else 0.0

    return stability


def compute_run_preservation(
    sub: np.ndarray,
    labels: np.ndarray,
    ref_labels: np.ndarray,
) -> dict:
    """Vectorised per-point preservation for a single clustering run.

    Parameters
    ----------
    sub : np.ndarray of shape (n_samples, k)
        Pre-computed kNN indices (self excluded).
    labels : np.ndarray of shape (n_samples,)
        Cluster labels for this run.
    ref_labels : np.ndarray of shape (n_samples,)
        Reference labels used for ARI computation.

    Returns
    -------
    dict with keys: pres (per-point), mean, ari, cover
    """
    labels = np.asarray(labels)
    ref_labels = np.asarray(ref_labels)
    n = len(labels)

    valid = labels >= 0
    neigh_labels = labels[sub]
    same = (neigh_labels == labels[:, None]) & valid[:, None]
    pres = same.sum(1) / sub.shape[1]

    ari, coverage = ari_core_weighted(ref_labels, labels)
    return {'pres': pres, 'mean': float(pres.mean()), 'ari': ari, 'cover': coverage}


def compute_stability_field(
    sub: np.ndarray,
    n: int,
    all_labels: list,
    ref_labels: np.ndarray,
) -> dict:
    """Vectorised per-point preservation across multiple clustering runs.

    Parameters
    ----------
    sub : np.ndarray of shape (n_samples, k)
        Pre-computed kNN indices (self excluded).
    n : int
        Number of samples (redundant but kept for API compatibility).
    all_labels : list of np.ndarray
        One label array per run (e.g. from :func:`~clusterlib.clustering.get_labels`).
    ref_labels : np.ndarray
        Reference labels for ARI computation.

    Returns
    -------
    dict with keys: mean (n,), std (n,), P (n_runs x n),
    ari (list), cover (list), runs (array of label arrays)
    """
    ref_labels = np.asarray(ref_labels)
    runs, P, ari_list, cover_list = [], [], [], []

    for labels in all_labels:
        labels = np.asarray(labels)
        valid = labels >= 0
        neigh_labels = labels[sub]
        same = (neigh_labels == labels[:, None]) & valid[:, None]
        pres = same.sum(1) / sub.shape[1]
        P.append(pres)
        a, cov = ari_core_weighted(ref_labels, labels)
        ari_list.append(a)
        cover_list.append(cov)
        runs.append(labels)

    P_mat = np.vstack(P)
    return {
        'mean':  P_mat.mean(0),
        'std':   P_mat.std(0),
        'P':     P_mat,
        'runs':  np.array(runs),
        'ari':   ari_list,
        'cover': cover_list,
    }




def stability_vs_k(
    k_values,
    df_features,
    ref_labels,
    labels,
    knn_metric: str = 'cosine',
) -> dict:
    """Per-point preservation for a *list* of neighbourhood sizes k.

    Builds a single kNN graph at max(k_values) and slices it for each k,
    comparing labels against ref_labels.

    Parameters
    ----------
    k_values : int or sequence of ints
    df_features : DataFrame or array
    ref_labels : array-like  -- reference clustering
    labels : array-like      -- clustering to evaluate
    knn_metric : str

    Returns
    -------
    dict keyed by k, each value is the output of :func:`compute_run_preservation`
    """
    if isinstance(k_values, int):
        k_values = [k_values]
    k_values = list(k_values)

    X = df_features.values if hasattr(df_features, 'values') else np.asarray(df_features)
    nbrs = NearestNeighbors(n_neighbors=max(k_values) + 1, metric=knn_metric).fit(X)
    _, idx = nbrs.kneighbors(X)
    idx = idx[:, 1:]  # exclude self

    out = {}
    for k in k_values:
        sub = idx[:, :k]
        out[k] = compute_run_preservation(sub, np.asarray(labels), np.asarray(ref_labels))
    return out


def grid_stability_vs_k(
    k_values,
    df_features,
    umap_grid: list[dict],
    hdbscan_grid: list[dict],
    ref_labels,
    knn_metric: str = 'cosine',
    seeds=(123,),
) -> dict:
    """Per-point preservation across a full parameter grid, for each k in *k_values*.

    Builds a single kNN graph at max(k_values) and runs every combination
    of UMAP / HDBSCAN parameters.

    Returns
    -------
    dict keyed by k, each value is the output of :func:`compute_stability_field`
    """
    if isinstance(k_values, int):
        k_values = [k_values]
    k_values = list(k_values)

    X = df_features.values if hasattr(df_features, 'values') else np.asarray(df_features)
    n = len(X)
    nbrs = NearestNeighbors(n_neighbors=max(k_values) + 1, metric=knn_metric).fit(X)
    _, idx = nbrs.kneighbors(X)
    idx = idx[:, 1:]

    all_labels = get_labels(X, umap_grid, hdbscan_grid, seeds)
    ref_labels = np.asarray(ref_labels)

    out = {}
    for k in k_values:
        sub = idx[:, :k]
        out[k] = compute_stability_field(sub, n, all_labels, ref_labels)
    return out


def characteristic_k(
    res_by_k: dict,
    thr: float = 0.5,
    field: str = 'mean',
) -> tuple[np.ndarray, np.ndarray]:
    """Per-point k at which stability first drops below *thr*.

    A high value indicates a deep, stable core; a low value indicates a
    boundary or loose cluster.

    Parameters
    ----------
    res_by_k : dict
        Output of :func: stability_vs_k or :func: grid_stability_vs_k.
    thr : float
        Stability threshold.
    field : str
        Which field to use ('mean' or 'std').

    Returns
    -------
    char_k : np.ndarray of shape (n_samples,)
    k_values : np.ndarray
    """
    k_values = np.array(sorted(res_by_k))
    S = np.vstack([res_by_k[int(k)][field] for k in k_values])
    n_points = S.shape[1]
    char_k = np.full(n_points, k_values[-1], dtype=float)
    hit = S < thr
    for i in range(n_points):
        w = np.where(hit[:, i])[0]
        if len(w):
            char_k[i] = k_values[w[0]]
    return char_k, k_values


def cluster_stability_summary(labels, stability) -> pd.DataFrame:
    """Aggregate per-point stability scores by cluster.

    Returns a DataFrame with columns mean, median, std, n
    sorted by mean stability descending.  Noise points are excluded from the
    table but their statistics are printed.
    """
    df = pd.DataFrame({'cluster': np.asarray(labels), 'stability': np.asarray(stability)})
    summary = (df[df['cluster'] >= 0]
               .groupby('cluster')['stability']
               .agg(mean='mean', median='median', std='std', n='count')
               .round(3)
               .sort_values('mean', ascending=False))

    noise_mask = np.asarray(labels) == -1
    if noise_mask.sum() > 0:
        noise_stab = np.asarray(stability)[noise_mask].mean()
        print(f"Noise points: {noise_mask.sum()}, mean stability: {noise_stab:.3f}")

    return summary



def bootstrap_stability(
    df_features,
    ref_labels,
    umap_params: dict,
    hdbscan_params: dict,
    frac: float = 0.8,
    n_iter: int = 20,
    k: int = 15,
    knn_metric: str = 'cosine',
    replace: bool = False,
    random_state=None,
) -> dict:
    """Estimate clustering stability by bootstrap subsampling.

    In each iteration a random subsample of *frac* × N points is drawn,
    clustered, and compared to *ref_labels* restricted to those points.
    Per-point preservation is averaged across all iterations in which the
    point was sampled.

    Returns
    -------
    dict with scalar summaries and per-point / per-iteration arrays.
    Compatible with :func:`~clusterlib.io.save_bootstrap`.
    """
    rng = np.random.default_rng(random_state)
    X = df_features.values if hasattr(df_features, 'values') else np.asarray(df_features)
    n = len(X)
    ref_labels = np.asarray(ref_labels)
    m = int(round(frac * n))

    aris, covers, means = [], [], []
    per_point_sum = np.zeros(n)
    per_point_cnt = np.zeros(n)

    for _ in range(n_iter):
        if replace:
            sample = rng.choice(n, size=m, replace=True)
            uniq = np.unique(sample)
        else:
            uniq = rng.choice(n, size=m, replace=False)

        sub_df = df_features.iloc[uniq].reset_index(drop=True)
        sub_labels = np.asarray(run_pipeline(sub_df, umap_params, hdbscan_params))

        nbrs = NearestNeighbors(n_neighbors=k, metric=knn_metric).fit(X[uniq])
        _, idx_local = nbrs.kneighbors(X[uniq])
        idx_local = idx_local[:, 1:]

        field = compute_run_preservation(idx_local, sub_labels, ref_labels[uniq])
        aris.append(field['ari'])
        covers.append(field['cover'])
        means.append(field['mean'])

        per_point_sum[uniq] += field['pres']
        per_point_cnt[uniq] += 1

    with np.errstate(invalid='ignore'):
        per_point_pres = np.where(per_point_cnt > 0,
                                  per_point_sum / per_point_cnt, np.nan)

    return {
        'stability_mean': float(np.mean(means)),
        'ari_mean':       float(np.mean(aris)),
        'ari_std':        float(np.std(aris)),
        'cover_mean':     float(np.mean(covers)),
        'pres_mean':      float(np.nanmean(per_point_pres)),
        'per_point_pres':  per_point_pres,
        'per_point_count': per_point_cnt,
        'ari':   np.array(aris),
        'cover': np.array(covers),
        'mean':  np.array(means),
    }


def seed_stability(
    df_features,
    ref_labels,
    umap_params: dict,
    hdbscan_params: dict,
    seeds=None,
    n_iter: int = 20,
    k: int = 15,
    knn_metric: str = 'cosine',
    random_state: int = 0,
) -> dict:
    """Measure stability across UMAP random seeds with fixed data.

    The kNN graph is built once and reused; only the UMAP random_state is
    varied.  The result has the same structure as :func: bootstrap_stability
    and is compatible with :func:~clusterlib.io.save_bootstrap.
    """
    rng = np.random.default_rng(random_state)
    X = df_features.values if hasattr(df_features, 'values') else np.asarray(df_features)
    n = len(X)
    ref_labels = np.asarray(ref_labels)

    if seeds is None:
        seeds = rng.integers(0, 2**31 - 1, size=n_iter)
    seeds = np.asarray(seeds)

    nbrs = NearestNeighbors(n_neighbors=k, metric=knn_metric).fit(X)
    _, idx = nbrs.kneighbors(X)
    idx = idx[:, 1:]

    aris, covers, means = [], [], []
    pres_stack = np.zeros((len(seeds), n))

    for i, seed in enumerate(seeds):
        up = {**umap_params, 'random_state': int(seed)}
        run_labels = np.asarray(run_pipeline(df_features, up, hdbscan_params))

        field = compute_run_preservation(idx, run_labels, ref_labels)
        aris.append(field['ari'])
        covers.append(field['cover'])
        means.append(field['mean'])
        pres_stack[i] = field['pres']

    per_point_pres = pres_stack.mean(0)

    return {
        'ari_mean':   float(np.mean(aris)),
        'ari_std':    float(np.std(aris)),
        'cover_mean': float(np.mean(covers)),
        'pres_mean':  float(np.mean(per_point_pres)),
        'per_point_pres':  per_point_pres,
        'per_point_count': np.full(n, len(seeds)),
        'ari':   np.array(aris),
        'cover': np.array(covers),
        'seeds': seeds,
    }




def per_point_stability(X, n_runs: int = 30, umap_params: dict | None = None,
                        hdbscan_params: dict | None = None):
    """Mode-based per-point stability: fraction of runs where a point ends up in
    its most-common cluster.

    Uses fixed default UMAP / HDBSCAN parameters if none are provided.
    """
    from collections import Counter
    import umap as _umap
    import hdbscan as _hdbscan

    if umap_params is None:
        umap_params = {'n_components': 15, 'min_dist': 0.1, 'n_neighbors': 30,
                       'metric': 'cosine'}
    if hdbscan_params is None:
        hdbscan_params = {'min_cluster_size': 10, 'min_samples': 20,
                          'metric': 'euclidean', 'cluster_selection_epsilon': 0.55,
                          'gen_min_span_tree': True}

    all_labels = []
    for seed in range(n_runs):
        emb = _umap.UMAP(random_state=seed, **umap_params).fit_transform(X)
        labels = _hdbscan.HDBSCAN(**hdbscan_params).fit_predict(emb)
        all_labels.append(labels)

    all_labels = np.array(all_labels)
    stability = np.zeros(len(X))
    for i in range(len(X)):
        col = all_labels[:, i]
        most_common_count = Counter(col).most_common(1)[0][1]
        stability[i] = most_common_count / n_runs

    return stability, all_labels


def per_point_stability_knn(X, umap_params: dict, hdbscan_params: dict,
                             n_runs: int = 30, k: int = 15) -> np.ndarray:
    """kNN co-occurrence stability: fraction of runs where a point and each of
    its k feature-space neighbours land in the same cluster.
    """
    import umap as _umap
    import hdbscan as _hdbscan

    n = len(X)
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    _, neighbor_idx = nbrs.kneighbors(X)

    cooccurrence = np.zeros((n, k))
    both_clustered = np.zeros((n, k))

    for seed in range(n_runs):
        emb = _umap.UMAP(**umap_params, random_state=seed).fit_transform(X)
        labels = _hdbscan.HDBSCAN(**hdbscan_params).fit_predict(emb)
        is_clustered = labels >= 0

        for i in range(n):
            nbr = neighbor_idx[i]
            both = is_clustered[i] & is_clustered[nbr]
            same = (labels[i] == labels[nbr]) & both
            both_clustered[i] += both
            cooccurrence[i] += same

    with np.errstate(invalid='ignore'):
        stability = np.where(
            both_clustered.sum(axis=1) > 0,
            cooccurrence.sum(axis=1) / both_clustered.sum(axis=1),
            0,
        )
    return stability


def per_point_stability_fast(X, umap_params: dict, hdbscan_params: dict,
                              n_runs: int = 30):
    """Vectorised co-occurrence stability using full pairwise co-occurrence matrix.

    More memory-intensive than :func: per_point_stability_knn but avoids the
    inner loop.  Returns (stability, co_rate) where co_rate is the full
    (n x n) co-occurrence rate matrix.
    """
    import umap as _umap
    import hdbscan as _hdbscan

    n = len(X)
    cooccurrence  = np.zeros((n, n))
    both_clustered = np.zeros((n, n))

    for seed in range(n_runs):
        emb = _umap.UMAP(**umap_params, random_state=seed).fit_transform(X)
        labels = _hdbscan.HDBSCAN(**hdbscan_params).fit_predict(emb)
        is_clustered = labels >= 0

        same_cluster = labels[:, None] == labels[None, :]
        both = is_clustered[:, None] & is_clustered[None, :]
        cooccurrence  += same_cluster & both
        both_clustered += both

    with np.errstate(invalid='ignore'):
        co_rate = np.where(both_clustered > 0,
                           cooccurrence / both_clustered, 0)

    stability = co_rate.sum(axis=1) / np.maximum(both_clustered.sum(axis=1), 1)
    return stability, co_rate


def point_stability(X, umap_params: dict, hdbscan_params: dict,
                    n_runs: int = 30) -> np.ndarray:
    """Per-point stability via full pairwise co-occurrence (loop version).

    Equivalent to :func:`per_point_stability_fast` but uses explicit loops —
    useful as a reference implementation.
    """
    import umap as _umap
    import hdbscan as _hdbscan

    n = len(X)
    cooccurrence  = np.zeros((n, n))
    both_clustered = np.zeros((n, n))

    for seed in range(n_runs):
        emb = _umap.UMAP(**umap_params, random_state=seed).fit_transform(X)
        labels = _hdbscan.HDBSCAN(**hdbscan_params).fit_predict(emb)

        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] >= 0 and labels[j] >= 0:
                    both_clustered[i, j] += 1
                    both_clustered[j, i] += 1
                    if labels[i] == labels[j]:
                        cooccurrence[i, j] += 1
                        cooccurrence[j, i] += 1

    stability = np.zeros(n)
    for i in range(n):
        if both_clustered[i].sum() > 0:
            stability[i] = cooccurrence[i].sum() / both_clustered[i].sum()

    return stability
