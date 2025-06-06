#Copyright (c) 2025 Evgeniia VOLCHOK
#for contacts e.p.volchok@gmail.com

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0

import os
import logging
from mclustering import*
import matplotlib.pyplot as plt
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
            extra_params={'file_to_read': default_filename} #{'file_to_write': default_filename}
        )
    
    logger.info('Features extracted successfully')

    print('Features are extracted/launched!')

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

    print('Clustering')

    types_clustering = {
        'dbscan': {'eps': 0.8, 'min_samples': 5, 'metric': 'euclidean'},
        'hdbscan': {},
        'kmeans': {}
    }
    df_features, _ = DBFuncs.split_into_two(processed)
    for type_cl, params_cl in types_clustering.items():
        clusters = Clustering(processed, results_dir, copy=True)
        logger.info(f'Clustering by {type_cl}')
        labels, num_clusters = clusters.do_clustering(df_features, model_type=type_cl, params=params_cl)
        print(f'Number of clusters {type_cl}: {num_clusters}')
        clusters.update_database()
        clusters.organize_files_by_cluster()

        logger.info(f'Saving database')
        file_to_write = default_filename+'_'+type_cl
        DBFuncs.save_database(clusters.df, file_to_write=file_to_write)

        logger.info(f'Visualization {type_cl}')
        print('Visualization dbscan')
        clusters.scores(clusters.df, labels)
        clusters.visualize(features.database, model_cluster=type_cl, params=params_cl)

    plt.show()
    


if __name__ == '__main__':
    
    log_name = Logg.get_log_filename()
    Logg.setup_logging(log_name)

    main()


