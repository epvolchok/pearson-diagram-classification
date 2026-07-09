#Copyright (c) 2026 Evgeniia VOLCHOK
#for contacts e.p.volchok@gmail.com

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree


def kernel_field(
    values: np.ndarray,
    X2D: np.ndarray,
    grid_res: int = 300,
    k: int = 40,
    bw=None,
    pad: float = 0.5,
) -> tuple:
    """Gaussian kernel-smoothed scalar field on a regular grid.

    Parameters
    ----------
    values : array of shape (n_samples,)
        Scalar value at each point (e.g. per-point stability).
    X2D : array of shape (n_samples, 2)
        2-D embedding coordinates.
    grid_res : int
        Grid resolution in each dimension.
    k : int
        Number of nearest neighbours used for smoothing.
    bw : float or None
        Gaussian bandwidth.  If None, set to the median distance to the 5th
        nearest neighbour.
    pad : float
        Extra padding added around the point cloud extents.

    Returns
    -------
    Xi, Yi : (grid_res, grid_res) meshgrids
    Zi : smoothed values on the grid
    Wi : sum of Gaussian weights (proxy for local density)
    """
    tree = cKDTree(X2D)
    if bw is None:
        d, _ = tree.query(X2D, k=6)
        bw = float(np.median(d[:, -1]))

    xi = np.linspace(X2D[:, 0].min() - pad, X2D[:, 0].max() + pad, grid_res)
    yi = np.linspace(X2D[:, 1].min() - pad, X2D[:, 1].max() + pad, grid_res)
    Xi, Yi = np.meshgrid(xi, yi)
    grid = np.column_stack([Xi.ravel(), Yi.ravel()])

    dist, idx = tree.query(grid, k=k)
    w = np.exp(-dist**2 / (2 * bw**2))
    Z = (w * values[idx]).sum(1) / w.sum(1)
    W = w.sum(1)
    return Xi, Yi, Z.reshape(Xi.shape), W.reshape(Xi.shape)



def plot_stability_contour(
    stability: np.ndarray,
    ax,
    fig,
    X2D: np.ndarray,
    title: str = 'Neighborhood preservation',
    count: bool = True,
) -> None:
    """Kernel-smoothed stability overlay using pcolormesh + contour lines.

    Transparency fades where data density is low, so the colour only shows
    in regions actually populated by points.

    Parameters
    ----------
    stability : array of shape (n_samples,)
        Per-point stability values in [0, 1].
    ax : matplotlib Axes
    fig : matplotlib Figure (needed for colorbar)
    X2D : array of shape (n_samples, 2)
    title : str
        Colorbar label.
    count : bool
        If True, overlay contour lines (masked to regions of high density).
    """
    Xi, Yi, Zi, Wi = kernel_field(stability, X2D)
    alpha = np.clip(Wi / (0.3 * Wi.max()), 0, 1) * 0.45

    mesh = ax.pcolormesh(Xi, Yi, Zi, cmap=plt.cm.RdYlGn,
                         vmin=0, vmax=1, shading='gouraud', zorder=1)
    mesh.set_alpha(alpha.ravel())

    if count:
        Zi_lines = np.ma.masked_where(Wi < 0.1 * Wi.max(), Zi)
        levels = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
        cs = ax.contour(Xi, Yi, Zi_lines, levels=levels,
                        colors='black', linewidths=0.6, alpha=0.5, zorder=2)
        ax.clabel(cs, fmt='%.2f', fontsize=7)

    levels = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(title)
    cbar.set_ticks(levels)
    cbar.set_ticklabels([str(l) for l in levels])
    ax.set_xlabel('UMAP-1 component')
    ax.set_ylabel('UMAP-2 component')


def plot_stability_rbf(
    stability: np.ndarray,
    ax,
    fig,
    X2D: np.ndarray,
    smoothing: float = 5.0,
    proximity_threshold: float = 0.5,
) -> None:
    """RBF-interpolated stability overlay with filled contours.

    Cells farther than proximity_threshold from any data point are masked so
    the contour does not bleed into empty regions.

    Parameters
    ----------
    stability : array of shape (n_samples,)
    ax, fig : matplotlib objects
    X2D : array of shape (n_samples, 2)
    smoothing : float
        RBF smoothing parameter (higher = smoother).
    proximity_threshold : float
        Maximum allowed distance from a data point before a grid cell is masked.
    """
    x, y = X2D[:, 0], X2D[:, 1]
    xi = np.linspace(x.min(), x.max(), 300)
    yi = np.linspace(y.min(), y.max(), 300)
    Xi, Yi = np.meshgrid(xi, yi)

    rbf = RBFInterpolator(
        np.column_stack([x, y]),
        stability,
        smoothing=smoothing,
        kernel='thin_plate_spline',
        neighbors=40,
    )
    Zi = rbf(np.column_stack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
    Zi = np.clip(Zi, 0, 1)

    tree = cKDTree(np.column_stack([x, y]))
    dist, _ = tree.query(np.column_stack([Xi.ravel(), Yi.ravel()]))
    dist = dist.reshape(Xi.shape)
    Zi_masked = np.ma.masked_where(dist > proximity_threshold, Zi)

    levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.75, 0.9, 1.0]
    cmap = plt.cm.RdYlGn

    cf = ax.contourf(Xi, Yi, Zi, levels=levels, cmap=cmap, alpha=0.35, zorder=1)
    cs = ax.contour(Xi, Yi, Zi, levels=levels,
                    colors='black', linewidths=0.6, alpha=0.5, zorder=2)
    ax.clabel(cs, fmt='%.2f', fontsize=7, inline=True)

    cbar = fig.colorbar(cf, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Neighborhood preservation', fontsize=11)
    cbar.set_ticks(levels)
    cbar.set_ticklabels([str(l) for l in levels])



def plot_ari_boxplot(boot: dict, ax=None, show_points: bool = True):
    """Boxplots of ARI and coverage from bootstrap / seed stability results.

    Draws ARI on the left y-axis and coverage on the right y-axis using two
    overlapping boxplots.

    Parameters
    ----------
    boot : dict
        Output of :func:~clusterlib.stability.bootstrap_stability or
        :func:~clusterlib.stability.seed_stability.
    show_points : bool
        If True, scatter individual run values over the boxes.

    Returns
    -------
    ax : matplotlib Axes (left axis)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 5))

    bp = ax.boxplot([boot['ari']], positions=[1], widths=0.6, patch_artist=True,
                    showmeans=True, meanline=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_alpha(0.4)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['ARI', 'coverage'])
    ax.set_ylim(max(0, boot['ari'].min() - 0.05), 1.01)
    ax.set_ylabel('ARI')
    ax.grid(axis='y', alpha=0.3)
    ax.set_title(f"ARI = {boot['ari_mean']:.2f} ± {boot['ari_std']:.2f}")
    ax.legend([bp['medians'][0], bp['means'][0]], ['median', 'mean'],
              loc='lower center')

    axR = ax.twinx()
    bp2 = axR.boxplot([boot['cover']], positions=[2], widths=0.6, patch_artist=True,
                      showmeans=True, meanline=True, showfliers=False)
    for patch in bp2['boxes']:
        patch.set_alpha(0.4)

    lo = float(boot['cover'].min()); hi = float(boot['cover'].max())
    pad = max((hi - lo) * 0.3, 0.005)
    axR.set_ylim(lo - pad, hi + pad)
    axR.set_ylabel('coverage')
    ax.set_xlim(0.5, 2.5)

    if show_points:
        rng = np.random.default_rng(0)
        ax.scatter(rng.normal(1, 0.04, len(boot['ari'])),
                   boot['ari'], s=18, alpha=0.6, color='k', zorder=3)
        axR.scatter(rng.normal(2, 0.04, len(boot['cover'])),
                    boot['cover'], s=18, alpha=0.6, color='k', zorder=3)
    return ax


def plot_preservation_overlay(
    emb2d: np.ndarray,
    per_point_pres: np.ndarray,
    ax=None,
    cmap: str = 'viridis',
    s: int = 12,
    nan_color: str = 'lightgray',
):
    """Scatter per-point preservation values on a 2-D embedding.

    Points that were never sampled (NaN) are shown in nan_color.

    Returns
    -------
    ax : matplotlib Axes
    """
    emb2d = np.asarray(emb2d)
    pres  = np.asarray(per_point_pres, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    nan = np.isnan(pres)
    if nan.any():
        ax.scatter(emb2d[nan, 0], emb2d[nan, 1], s=s, c=nan_color,
                   label='never sampled', zorder=1)
    sc = ax.scatter(emb2d[~nan, 0], emb2d[~nan, 1], c=pres[~nan],
                    s=s, cmap=cmap, vmin=0, vmax=1, zorder=2)
    plt.colorbar(sc, ax=ax, label='preservation')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2')
    ax.set_title('Per-point stability on embedding')
    if nan.any():
        ax.legend(loc='upper right', fontsize=8)
    return ax


def plot_stability_report(
    boot: dict,
    emb2d: np.ndarray,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """Two-panel stability figure: ARI boxplot + preservation contour map.

    Parameters
    ----------
    boot : dict
        Output of :func:~clusterlib.stability.bootstrap_stability or
        :func:~clusterlib.stability.seed_stability.
    emb2d : array of shape (n_samples, 2)
        2-D embedding for the contour map.
    figsize : tuple

    Returns
    -------
    fig : matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                   gridspec_kw={'width_ratios': [1, 1.4]})
    plot_ari_boxplot(boot, ax=ax1)
    plot_stability_contour(boot['per_point_pres'], ax2, fig, emb2d,
                           title='Neighborhood preservation', count=True)
    plt.tight_layout()
    return fig




def plot_clusters(
    ax,
    X2D: np.ndarray,
    labels,
    palette=None,
    noise_color: str = 'gray',
    s: int = 10,
) -> None:
    """Scatter plot of a 2-D embedding coloured by cluster labels.

    Parameters
    ----------
    ax : matplotlib Axes
    X2D : array of shape (n_samples, 2)
    labels : array-like of ints (−1 for noise)
    palette : list of colours or None
        If None, uses matplotlib's tab10.
    noise_color : str
        Colour for noise points (label == -1).
    s : int
        Marker size.
    """
    if palette is None:
        palette = plt.get_cmap('tab10')
        palette = [palette(i % 10) for i in range(20)]

    labels = np.asarray(labels)
    for label in sorted(set(labels)):
        mask = labels == label
        color = noise_color if label == -1 else palette[label % len(palette)]
        lbl = 'Noise' if label == -1 else str(label)
        ax.scatter(X2D[mask, 0], X2D[mask, 1], s=s, color=color, label=lbl)

    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')
