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
from matplotlib import gridspec
logger = logging.getLogger(__name__)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



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
    filter_mixed = False
    features = ResNetFeatures(
            path=os.path.join(cwd, 'images', input_imags),
            flag=flag,
            info_path=info_path,
            filter_mixed=filter_mixed,
            name_pattern=name_pattern,
            extra_params={'file_to_read': default_filename+'_features'} #{'file_to_read': default_filename}
        )
    
    logger.info('Features extracted successfully')

    print('Features are extracted/launched!')

    ResNetFeatures.info_on_features(features)


    print('Filtration')
    logger.info(f'Filtration')
    features.database = features.filtering_by_variance()
    ResNetFeatures.info_on_features(features)

    logger.info(f'Saving database')
    file_to_write = default_filename+'_filtered'
    DBFuncs.save_database(features.database, file_to_write=file_to_write)

    df_features, info_db = DBFuncs.split_into_two(features.database)


    grs = GridSearch(df_features, copy=True)

    best_silhouette, best_db = grs.search_params(pca_flag=False)
    grs.save_pipe(best_silhouette, 'best_silhouette_without_pca')
    grs.save_pipe(best_db, 'best_db_without_pca')

    best_silhouette_pca, best_db_pca = grs.search_params(pca_flag=True)
    grs.save_pipe(best_silhouette_pca, 'best_silhouette_with_pca')
    grs.save_pipe(best_db_pca, 'best_db_with_pca')



    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(16, 9))

    grs.visualize_best_score(axs, best_silhouette, best_db, pca_flag=False)

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(16, 9))

    grs.visualize_best_score(axs, best_silhouette_pca, best_db_pca, pca_flag=True)

    plt.show()
    



if __name__ == '__main__':
    
    log_name = Logg.get_log_filename()
    Logg.setup_logging(log_name)

    main()


