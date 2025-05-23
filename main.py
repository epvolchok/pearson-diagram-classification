#Copyright (c) 2025 Evgeniia VOLCHOK
#for contacts e.p.volchok@gmail.com

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0

import os
import logging
from mclusterization import*

logger = logging.getLogger(__name__)

def main():

    cwd = os.getcwd()
    base_dir_names = ['images', 'processed', 'figures', 'data', 'results', 'logs']

    for dirname in base_dir_names:
        dirname = os.path.join(cwd, dirname)
        ServiceFuncs.preparing_folder(dirname, clear=False)
    logger.info('Base directories checked or created')

    input_imags = 'images_reg_b'
    results_dir = os.path.join(cwd, 'processed', 'processed_reg_b')
    ServiceFuncs.preparing_folder(results_dir, clear=True)
    default_filename = os.path.join(cwd, 'results', 'pearson_diagram_data_reg_b')
    logger.info(f'Source images: {input_imags} Results: {results_dir, default_filename}')

    info_path = os.path.join(cwd, 'data', 'SOLO_info_rswf.txt')
    name_pattern = r'(solo_L2_rpw-tds-surv-(?:r|t)swf-e_\d+\w+)'

    flag = 'read' #or extract
    filter_mixed = True
    features = ResNetFeatures(
            path=os.path.join(cwd, 'images', input_imags),
            flag=flag,
            info_path=info_path,
            filter_mixed=filter_mixed,
            name_pattern=name_pattern,
            extra_params={'file_to_write': default_filename}
        )
    
    logger.info('Features extracted successfully')

    print('Features are extracted/launched!')

    message = 'The standard algorithm: \n' + \
            '1. Filtration of ResNet features by variance, thershold=1e-5 \n' + \
            '2. Preprocessing of the filtered features with \n' + \
            '   - PCA(n_components=0.95, svd_solver=\'full\') \n' + \
            '   - UMAP(n_components=20, min_dist=0.1, metric=\'cosine\') \n' + \
            '3. Clusterization \n' + \
            '- HDBSCAN(min_cluster_size=15, min_samples=5, metric=\'euclidean\') \n' + \
            '4. Visualization with PCA+UMAP2D+HDBSCAN'
    print(message)
    logger.info('Standard algorithm launched')

    print('Filtration')
    logger.info(f'Filtration')
    features.database = features.filtering_by_variance()
    features.info_on_features()

    logger.info(f'Saving database')
    file_to_write = default_filename+'_filtered'
    DBFuncs.save_database(features.database, file_to_write=file_to_write)

    print('Preprocessing')
    logger.info('Preprocessing')
    preprop = FeaturesPreprocessing(features)
    processed = preprop.wrapper_preprop(features.database,'PCA+UMAPND')

    print('Clusterization')
    logger.info('Clusterization')
    clusters = Clustering(processed)
    df_features, _ = DBFuncs.split_into_two(processed)
    _, num_clusters = clusters.clustering_HDBSCAN(df_features)
    print(f'Number of clusters 20D: {num_clusters}')
    clusters.update_database()
    clusters.sort_files()

    logger.info(f'Saving database')
    file_to_write = default_filename+'_clustered'
    DBFuncs.save_database(clusters.df, file_to_write=file_to_write)

    logger.info('Visualization')
    clusters.visualize_HDBSCAN(features.database)


if __name__ == '__main__':
    
    log_name = Logg.get_log_filename()
    Logg.setup_logging(log_name)

    main()


