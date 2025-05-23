#Copyright (c) 2025 Evgeniia VOLCHOK
#for contacts e.p.volchok@gmail.com

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0

import os
import pandas as pd
from mclustering import*
from .libinteractive import InputManager, PathManager

import logging
logger = logging.getLogger(__name__)

class FeatureManager:
    """
    This class serves as a high-level command-line wrapper around the functionality of `libfeatures`.
    It rules the process of features extraction from images or loading from a file.

    This class is not intended to be instantiated.
    """
    def __init__(self):
        ServiceFuncs.init_error(__class__.__name__)
    
    @staticmethod
    def get_features(input_imags: str, info_path: str, default_filename: str, name_pattern: str):
        """
        Asks the user whether to extract new features or read from file.

        Parameters
        ----------
        input_imags : str
            Subdirectory name with image data.
        info_path : str
            Path to the metadata file.
        default_filename : str
            Default base filename for feature data.
        name_pattern : str
            Regex pattern for identifying dataset names.

        Returns
        -------
        ResNetFeatures
            An instance of the ResNetFeatures class containing the loaded/extracted features.
        """
        message = f'To extract features from images located in {input_imags} enter 1. \n' + \
                'To load features from a file enter 2.'
        print(message)
        logger.info('Getting features')
        
        while True:
            choice = InputManager.input_wrapper('[1/2] > ').strip()
            logger.info(f'Choice: {choice}')
            if choice == '1':
                features = FeatureManager.extract_features(input_imags, info_path, default_filename, name_pattern)
                break
            elif choice == '2':
                features = FeatureManager.read_features(input_imags, info_path, default_filename, name_pattern)
                break
            elif choice == 'break':
                break
            else:
                print('Invalid input. \n Try again!')
                logger.debug('Invalid input. Please enter 1 or 2 (or break).')
        return features

    @staticmethod
    def extract_features(input_imags, info_path, default_filename, name_pattern):
        """
        Initializes feature extraction and saves them to file.

        Parameters
        ----------
        input_imags : str
            Folder containing images.
        info_path : str
            Path to the metadata file.
        default_filename : str
            Default file path for saving results.
        name_pattern : str
            Regex pattern for dataset name recognition.

        Returns
        -------
        ResNetFeatures
            Object containing the extracted feature database.
        """
        message = f'Enter a file name to write extracted features (or press "Enter" for default "{default_filename}") > '
        file_to_write = PathManager.get_path(message, default_filename)
        filter_mixed, name_pattern = InputManager.filter_mixed_freq(name_pattern)
        cwd = os.getcwd()

        print('Features extraction')
        logger.info(f'Features extraction. File to write: {file_to_write}.')

        features = ResNetFeatures(
            path=os.path.join(cwd, 'images', input_imags),
            flag='extract',
            info_path=info_path,
            filter_mixed=filter_mixed,
            name_pattern=name_pattern,
            extra_params={'file_to_write': file_to_write}
        )
        logger.info('Features extracted successfully')
        return features
    
    @staticmethod
    def read_features(input_imags, info_path, default_filename, name_pattern):
        """
        Loads previously saved feature data from a file.

        Parameters
        ----------
        input_imags : str
            Folder containing images.
        info_path : str
            Path to the metadata file.
        default_filename : str
            Default file path for loading.
        name_pattern : str
            Regex pattern for dataset name recognition.

        Returns
        -------
        ResNetFeatures
            Object containing the loaded feature database.
        """

        message = f'Enter a file name to read from (or press "Enter" for default "{default_filename}") > '
        file_to_read = PathManager.get_path(message, default_filename)
        filter_mixed, name_pattern = InputManager.filter_mixed_freq(name_pattern)
        cwd = os.getcwd()

        print('Loading features')
        logger.info(f'Loading features. File to read from: {file_to_read}.')

        features = ResNetFeatures(
            path=os.path.join(cwd, 'images', input_imags),
            flag='read',
            info_path=info_path,
            filter_mixed=filter_mixed,
            name_pattern=name_pattern,
            extra_params={'file_to_read': file_to_read}
        )
        logger.info('Features loaded successfully')
        return features

class ProcessingPipeline:
    
    def __init__(self):
        ServiceFuncs.init_error(__class__.__name__)

    @staticmethod
    def run_processing(features, default_filename):
        """
        Offers the user a choice between standard pipeline execution or custom steps.

        Parameters
        ----------
        features : ResNetFeatures
            Feature database.
        default_filename : str
            Base name for saving intermediate or final results.
        """
        message = 'Would you like to launch the standard processing algorithm (1) or \n' + \
                    'to call separate blocks of processing manually (2)?'
        print(message)
        logger.info('Processing running')
        while True:
            choice = InputManager.input_wrapper('[1/2] > ').strip()
            logger.info(f'Choice: {choice}')
            if choice == '1':
                ProcessingPipeline.standard_algorithm(features, default_filename)
            elif choice == '2':
                ProcessingPipeline.choose_block(features, default_filename)
            elif choice == 'break':
                break
            else:
                print('Invalid input. \n Try again!')
                logger.debug('Invalid input. Please enter 1 or 2 (or break).')



    @staticmethod
    def standard_algorithm(features, default_filename):
        """
        Executes the full default processing pipeline:
        1. Feature filtering by variance
        2. Dimensionality reduction with PCA and UMAP
        3. Clustering with HDBSCAN
        4. Visualization

        Parameters
        ----------
        features : ResNetFeatures
            Input feature database object.
        default_filename : str
            Base filename for saving outputs.
        """
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

        features = ProcessingPipeline.filtration(features)

        message = f'Enter a file name to save the filtered data (or press "Enter" to use default name) > '
        PathManager.saving_database(features.database, default_filename, message, suf='_filtered')

        processed = ProcessingPipeline.run_preprocessing(features, 'PCA+UMAPND')

        clusters = ProcessingPipeline.run_clusterization(processed)

        message = 'Enter a file name to save the clustered data (or press "Enter" to use default name) > '
        PathManager.saving_database(clusters.df, default_filename, message, suf='_clustered')

        # Visualization
        logger.info('Visualization')
        clusters.visualize_HDBSCAN(features.database)

    @staticmethod
    def filtration(features):
        """
        Applies variance-based filtering to the feature set.

        Parameters
        ----------
        features : ResNetFeatures
            Input feature database.

        Returns
        -------
        ResNetFeatures
            Filtered features with updated `.database`.
        """
        print('Filtration')
        logger.info(f'Filtration')
        features.database = features.filtering_by_variance()
        features.info_on_features()
        return features
    
    @staticmethod
    def run_preprocessing(features, pipe_str='PCA+UMAPND', params={}):
        """
        Applies a sequence of dimensionality reduction steps to the feature matrix.

        Parameters
        ----------
        features : ResNetFeatures
            The extracted and optionally filtered features.
        pipe_str : str, optional
            Pipeline specification string, e.g., 'PCA+UMAPND'.

        Returns
        -------
        pandas.DataFrame
            Transformed feature DataFrame with original metadata preserved.   
        """
        print('Preprocessing')
        logger.info('Preprocessing')
        preprop = FeaturesPreprocessing(features)
        if isinstance(features, ResNetFeatures):
            df = features.database
        elif isinstance(features, pd.DataFrame):
            df = features
        else:
            logger.error(ValueError('(features) is non-known object'))
            raise ValueError('(features) is non-known object')
        processed = preprop.wrapper_preprop(df, pipe_str, params)
        return processed
    
    @staticmethod
    def run_clusterization(features):
        """
        Performs HDBSCAN clustering and sorts image files into corresponding folders.

        Parameters
        ----------
        features : pandas.DataFrame
            The feature matrix with metadata.

        Returns
        -------
        Clustering
            Clustering object with updated label assignments and organized folders.
        """

        print('Clusterization')
        logger.info('Clusterization')
        clusters = Clustering(features)
        df_features, _ = DBFuncs.split_into_two(features)
        _, num_clusters = clusters.clustering_HDBSCAN(df_features)
        print(f'Number of clusters 20D: {num_clusters}')
        clusters.update_database()
        clusters.organize_files_by_cluster()
        return clusters
    
    @staticmethod
    def choose_block(features, default_filename):

        message = 'Choose a block of data processing: \n' + \
        '1. preprocessing: scaling and dimensions reduction \n' + \
        '2. clusterization of processed data (compress dimensions first!) \n' + \
        '3. analysis of obtained results (haven\'t implemented)\n' + \
        'or enter "break" to return to the higher level'

        print(message)

        while True:
            choice = InputManager.input_wrapper('[1/2/3] >  ').strip()
            logger.info(f'Choice: {choice}')
            if choice == '1':
                ProcessingPipeline.preprocessing_setup(features, default_filename)
            elif choice == '2':
                print('This part has not been implemented yet. Use 1 or break.')
            elif choice == '3':
                print('This part has not been implemented yet. Use 1 or break.')
            elif choice == 'break':
                break
            else:
                print('Invalid input. \n Please enter 1, 2 or 3 (or break).')
                logger.debug('Invalid input.')

    @staticmethod
    def preprocessing_setup(features, default_filename):
        message = 'To preprocess, enter a pipeline in the form \n' + \
        '"pca+umapnd+<...>". \n' + \
        'You can choose any (reasonable) combinations of \n' + \
        'sklearn.Normalizer() - use "normalizer" \n' + \
        'sklearn.StandartScalar() - use "scalar" \n' + \
        'sklearn.PCA() - use "pca" \n' + \
        'umap.UMAP() - use "umapnd". \n' + \
        'You will be able to set parameters for any of these funcs (enter "params") \n' + \
        'or use default parameters: \n' + \
        'default parameters: \n' + \
        f'{FeaturesPreprocessing(features).default_params}.'

        print(message)

        while True:
            pipeline = InputManager.input_wrapper('Enter a pipeline: ').strip()
            logger.info(f'Pipeline: {pipeline}')
            if pipeline == 'break':
                break

            if InputManager.get_bool('Would you like to enter params? '):
                params = ProcessingPipeline.get_processing_params(pipeline)
            else:
                params = {}
            processed = ProcessingPipeline.run_preprocessing(features, pipeline, params)

            ProcessingPipeline.to_deal_results(pipeline, processed, default_filename)

    @staticmethod
    def to_deal_results(pipeline, processed, default_filename):

        while True:
            choice = InputManager.input_wrapper('Enter "save", "print", "break" or "continue" to try again > ').strip()
            logger.info(f'Choice: {choice}')
            if choice == 'save':
                message = f'Enter a file name to save data (or press "Enter" to use default_name: {default_filename}) > '
                PathManager.saving_database(processed, default_filename, message, suf='_processed')
            elif choice == 'print':
                visualize = InputManager.get_bool('Would you like to plot histogram of features varience? > ')
                ResNetFeatures.info_on_features(processed, visualize, title=pipeline)
            elif choice == 'break':
                break
            else:
                print('Invalid input. \n Try again!')
                logger.debug('Invalid input.')

    @staticmethod
    def get_processing_params(pipeline):
        print(f'Enter parameters for every or some of steps in {pipeline} >')
        models = pipeline.lower().split('+')
        params = {}
        for name in models:
            param = InputManager.input_dict(f'Enter parameters for {name} (as key=value) > ')
            if param:
                params[name] = param
        return params