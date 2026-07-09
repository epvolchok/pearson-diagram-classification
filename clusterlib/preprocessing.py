#Copyright (c) 2026 Evgeniia VOLCHOK
#for contacts e.p.volchok@gmail.com

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold

import logging
logger = logging.getLogger(__name__)


def variance_filter(df: pd.DataFrame, threshold: float = 5e-5) -> pd.DataFrame:
    """Remove features whose variance falls below threshold.

    Returns a DataFrame with the same index but only columns that pass the
    variance threshold filter.
    """
    logger.info(f"Filtration on variance. Thershold {threshold}")
    selector = VarianceThreshold(threshold=threshold)
    filtered = selector.fit_transform(df)

    logger.info(f"After variance filtration {filtered.shape[1]}")
    return pd.DataFrame(filtered, index=df.index,
                        columns=selector.get_feature_names_out())


def remove_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.9,
    corr_method: str = 'pearson',
) -> tuple[list, set]:
    """Single-pass removal of one feature from each highly correlated pair.

    For each pair with |correlation| > threshold, removes the feature that
    has the higher count of correlated partners (tie-break: higher mean
    pairwise correlation).

    Returns
    -------
    to_keep : list of column names that survive
    to_drop : set of column names that were dropped
    """
    corrmax = df.corr(method=corr_method).abs()
    n = len(corrmax)
    mean_corr = (corrmax.sum(axis=1) - 1) / (n - 1)
    high_corr_count = (corrmax > threshold).sum(axis=1) - 1

    upper_tri = corrmax.where(
        np.triu(np.ones(corrmax.shape), k=1).astype(bool)
    )
    pairs = (
        upper_tri.stack()
                 .reset_index()
                 .rename(columns={'level_0': 'feature_1', 'level_1': 'feature_2', 0: 'corr'})
                 .query('corr > @threshold')
                 .sort_values('corr', ascending=False)
    )

    to_drop = set()
    for _, row in pairs.iterrows():
        f1, f2 = row['feature_1'], row['feature_2']
        count1, count2 = high_corr_count[f1], high_corr_count[f2]
        mean1,  mean2  = mean_corr[f1],       mean_corr[f2]

        if count1 != count2:
            loser = f1 if count1 > count2 else f2
        else:
            loser = f1 if mean1 >= mean2 else f2
        to_drop.add(loser)

    to_keep = [col for col in corrmax.columns if col not in to_drop]
    return to_keep, to_drop


def filter_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.85,
    corr_method: str = 'pearson',
    verbose: bool = False,
) -> pd.DataFrame:
    """Iterative removal of correlated features until no pair exceeds threshold.

    Unlike :func:remove_correlated_features, recomputes the correlation
    matrix after each round so that cascading redundancies are resolved
    correctly.

    Returns a DataFrame containing only the surviving columns.
    """
    logger.info('Filtration on correlation')
    remaining = list(df.columns)
    iteration = 0

    while True:
        iteration += 1
        corr_matrix = df[remaining].corr(method=corr_method).abs()
        n = len(remaining)
        mean_corr       = (corr_matrix.sum(axis=1) - 1) / (n - 1)
        high_corr_count = (corr_matrix > threshold).sum(axis=1) - 1

        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        pairs = (
            upper.stack()
                 .reset_index()
                 .rename(columns={'level_0': 'feature_1', 'level_1': 'feature_2', 0: 'corr'})
                 .query('corr > @threshold')
                 .sort_values('corr', ascending=False)
        )

        if pairs.empty:
            break

        dropped_this_round: set = set()
        for _, row in pairs.iterrows():
            f1, f2 = row['feature_1'], row['feature_2']
            if f1 in dropped_this_round or f2 in dropped_this_round:
                continue
            count1, count2 = high_corr_count[f1], high_corr_count[f2]
            mean1,  mean2  = mean_corr[f1],       mean_corr[f2]
            loser = f1 if (count1 > count2 or (count1 == count2 and mean1 >= mean2)) else f2
            dropped_this_round.add(loser)

        for f in dropped_this_round:
            remaining.remove(f)

        if verbose:
            print(f"Iteration {iteration}: dropped {len(dropped_this_round)}, "
                  f"remaining {len(remaining)}")
        logger.info(f"Iteration {iteration}: dropped {len(dropped_this_round)}, "
                  f"remaining {len(remaining)}")
    return df[remaining]


def plot_correlation_threshold(
    df: pd.DataFrame,
    thresholds: list | None = None,
    ax=None,
) -> None:
    """Plot number of retained features vs. correlation threshold.

    Useful for choosing a threshold before calling
    :func:filter_correlated_features.
    """
    if thresholds is None:
        thresholds = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]

    num_feat = []
    for th in thresholds:
        filtered = filter_correlated_features(df, threshold=th)
        num_feat.append(filtered.shape[1])

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    ax.plot(thresholds, num_feat, marker='o')
    ax.set_xlabel('Correlation threshold')
    ax.set_ylabel('Features retained')
    ax.grid(alpha=0.3)
