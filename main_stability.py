#Copyright (c) 2026 Evgeniia VOLCHOK
#for contacts e.p.volchok@gmail.com

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0

import os
from itertools import product
import logging
from mclustering import*

import matplotlib
matplotlib.use('qtagg')
from matplotlib import rc
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from clusterlib.preprocessing import variance_filter, filter_correlated_features
from clusterlib.clustering    import run_pipeline, embed_2d, n_clusters
from clusterlib.stability     import grid_stability_vs_k, label_preservation
from clusterlib.visualization import plot_stability_contour
from clusterlib.io            import save_stability_vs_k, load_stability_vs_k

logger = logging.getLogger(__name__)

rc('text', usetex=True)
rc('font', family='serif')
rc('text.latex', preamble=r'\usepackage[utf8]{inputenc}')
matplotlib.rcParams.update({'font.size': 16})


# Reference parameters for clustering 
# and dimension reduction
ref_umap_params = {
    'n_components': 30,
    'min_dist': 0.1,
    'n_neighbors': 20,
}
ref_hdbscan_params = {
    'min_cluster_size': 10,
    'min_samples': 20,
    'metric': 'euclidean',
    'cluster_selection_epsilon': 0.6,
    'cluster_selection_method': 'leaf',
}

# Parameters for grid search
# and stability study
umap_grid = [
    {'n_components': nc, 'n_neighbors': nn, 'min_dist': md, 'metric': 'cosine'}
    for nc, nn, md in product([10, 20, 30], [10, 20, 30, 40], [0.1, 0.2, 0.3])
]
hdbscan_grid = [
    {
        'min_cluster_size':         mcs,
        'min_samples':              ms,
        'cluster_selection_epsilon': eps,
        'metric':                   'euclidean',
    }
    for mcs, ms, eps in product(
        [10, 15, 20, 30],
        [5, 10, 20, 30, 40],
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    )
    if ms <= 2 * mcs
]


def reference_clustering(df_pca):
    """
    Reference clustering results
    """
    ref_labels = run_pipeline(
        df_pca,
        umap_params=ref_umap_params,
        hdbscan_params=ref_hdbscan_params,
        random_state=123,
    )
    k = n_clusters(ref_labels)
    noise_frac = (ref_labels == -1).mean()
    logger.info(f"{k} clusters, "
          f"{noise_frac:.1%} noise ({(ref_labels == -1).sum()} pts)")
    return ref_labels, k, noise_frac



def compute_or_load_stability(df_pca, ref_labels, k_stability, path,
                               force_recompute=False):
    """
    Grid stability study
    """
    npz_path = path if path.endswith('.npz') else path + '.npz'

    if os.path.exists(npz_path) and not force_recompute:
        logger.info(f"Loading precomputed stability from {npz_path}")
        return load_stability_vs_k(npz_path)

    logger.info(f"Running grid stability study"
                   f"k = {k_stability}")
    res_by_k = grid_stability_vs_k(
        k_values    = [k_stability],
        df_features = df_pca,
        umap_grid   = umap_grid,
        hdbscan_grid= hdbscan_grid,
        ref_labels  = ref_labels,
        knn_metric  = 'cosine',
        seeds       = (123,),
    )
    save_stability_vs_k(res_by_k, path)
    return res_by_k


def per_point_reference_agreement(res_by_k, ref_labels, k):
    """
    res_by_k[k]['mean']  is per-point neighbourhood self-consistency across
    all grid runs. For the ARI panel we want per-point agreement with the
    reference clustering: for each point, the fraction of grid runs in which
    its cluster neighbourhood matches the reference neighbourhood.
    We compute this via label_preservation(ref, run) averaged over runs.
    """
    all_run_labels = res_by_k[k]['runs']   # shape (n_runs, n_points)
    agreements = [
        label_preservation(ref_labels, run_labels)
        for run_labels in all_run_labels
    ]
    return np.mean(agreements, axis=0)



def plot_results(res_by_k_global, X_umap2D, stability_mean, ref_agreement, k_stability, 
                 ref_labels, fig_path='./figures/stability_main.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Background scatter coloured by reference clusters
    unique_labels = sorted(set(ref_labels))
    cmap = plt.get_cmap('tab10')
    for label in unique_labels:
        mask = ref_labels == label
        color = 'lightgray' if label == -1 else cmap(label % 10)
        ax1.scatter(X_umap2D[mask, 0], X_umap2D[mask, 1],
                    s=6, color=color, zorder=0)
        ax2.scatter(X_umap2D[mask, 0], X_umap2D[mask, 1],
                    s=6, color=color, zorder=0)

    # Left panel – neighbourhood preservation across grid
    plot_stability_contour(
        stability  = stability_mean,
        ax         = ax1,
        fig        = fig,
        X2D        = X_umap2D,
        title      = r'Neighbourhood preservation',
        count      = True,
    )
    ax1.set_title(r'Grid stability (self-consistency)')

    # Right panel – agreement with the reference clustering
    plot_stability_contour(
        stability  = ref_agreement,
        ax         = ax2,
        fig        = fig,
        X2D        = X_umap2D,
        title      = r'Agreement with reference',
        count      = True,
    )
    mean_ari  = np.mean(res_by_k_global[k_stability]['ari'])
    std_ari   = np.std(res_by_k_global[k_stability]['ari'])
    ax2.set_title(rf'ARI vs reference: {mean_ari:.2f} $\pm$ {std_ari:.2f}')

    plt.tight_layout()
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=300, format='png')
    print(f"Figure saved to {fig_path}")
    plt.show()




def main():

    cwd = os.getcwd()
    base_dir_names = ['images', 'processed', 'figures', 'data', 'results', 'logs']

    for dirname in base_dir_names:
        dirname = os.path.join(cwd, dirname)
        ServiceFuncs.preparing_folder(dirname, clear=False)
    logger.info('Base directories checked or created')

    info_path = os.path.join(cwd, 'data', 'SOLO_info_rswf.txt')
    name_pattern = r'(solo_L2_rpw-tds-surv-(?:r|t)swf-e_\d+\w+)'
    input_imags = 'images_all_reg'
    default_filename = os.path.join(cwd, 'results', 'pearson_diagram_all_reg')

    flag = 'read' #or extract
    filter_mixed = False
    features = ResNetFeatures(
            path=os.path.join(cwd, 'images', input_imags),
            flag=flag,
            info_path=info_path,
            filter_mixed=filter_mixed,
            name_pattern=name_pattern,
            extra_params={'file_to_read': default_filename+'_features'}
        )
    logger.info('Features read successfully')

    ResNetFeatures.info_on_features(features)

    print('Filtration')
    excluded_columns = ['dataset_name', 'date', 'dist_to_sun[au]',
    'SAMPLES_NUMBER', 'SAMPLING_RATE[kHz]', 'SAMPLE_LENGTH[ms]', 'oldpath']
    df_features, excluded_part = DBFuncs.split_into_two(features.database, excluded_columns=excluded_columns)
    df_corr = filter_correlated_features(df_features, threshold=0.85, verbose=True)
    final_df = pd.concat([excluded_part, df_corr], axis=1)
    features.database = final_df
    features.database = features.filtering_by_variance()
    ResNetFeatures.info_on_features(features)
    print(f"After variance filter: {features.database.shape[1]}")

    logger.info(f'Saving database')
    file_to_write = default_filename+'_filtered'
    DBFuncs.save_database(features.database, file_to_write=file_to_write)

    print('Preprocessing')
    logger.info('Preprocessing')
    preprop = FeaturesPreprocessing(features)
    processed = preprop.wrapper_preprop(features.database,'PCA')


    print("Reference clustering")
    logger.info("Reference clustering")
    df_features, _ = DBFuncs.split_into_two(processed, excluded_columns)
    ref_labels, k, noise_frac = reference_clustering(df_features)
    print(f"{k} clusters, "
          f"{noise_frac:.1%} noise ({(ref_labels == -1).sum()} pts)")

    k_stability = 15
    print(f"Grid stability, k: {k_stability}")
    path = os.path.join(cwd, 'results', default_filename + '_stability')
    res_by_k = compute_or_load_stability(df_features, ref_labels, k_stability, path)
    res_by_k_global = res_by_k
    logger.info(f"stability result: {res_by_k}")
    
    stability_mean = res_by_k[k_stability]['mean']
    ref_agreement  = per_point_reference_agreement(res_by_k, ref_labels, k_stability)

    mean_ari = np.mean(res_by_k[k_stability]['ari'])
    print(f"Grid ARI vs reference:  mean = {mean_ari:.3f}, "
          f"std = {np.std(res_by_k[k_stability]['ari']):.3f}")
    print(f"Mean neighbourhood preservation: "
          f"{stability_mean.mean():.3f}")
    print(f"Mean reference agreement: "
          f"{ref_agreement.mean():.3f}")

    
    print("Visualisation")
    logger.info("Visualisation")
    vis_umap_params = {
        'min_dist': ref_umap_params['min_dist'],
        'n_neighbors': ref_umap_params['n_neighbors']}
    X_umap2D = embed_2d(df_features, vis_umap_params, random_state=42)

    plot_results(res_by_k_global, X_umap2D, stability_mean, ref_agreement, k_stability, 
                 ref_labels)


if __name__ == '__main__':
    log_name = Logg.get_log_filename()
    Logg.setup_logging(log_name)
    main()
