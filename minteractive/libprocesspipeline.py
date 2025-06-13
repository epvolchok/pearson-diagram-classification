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
from mclustering import DBFuncs
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)
from typing import Optional, Dict, Tuple, Any

class FeatureManager:
    """
    This class serves as a high-level command-line wrapper around the functionality of `libfeatures`.
    It rules the process of features extraction from images or loading from a file.

    This class is not intended to be instantiated.
    """
    def __init__(self):
        ServiceFuncs.init_error(__class__.__name__)
    
    @staticmethod
    def get_features(input_imags: str, info_path: str, default_filename: str, name_pattern: str)-> ResNetFeatures:
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
                print('Invalid input. \n Try again! > ')
                logger.debug('Invalid input. Please enter 1 or 2 (or break).')
        return features

    @staticmethod
    def extract_features(input_imags: str, info_path: str, default_filename:str, name_pattern: str)-> ResNetFeatures:
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
    def read_features(input_imags: str, info_path: str, default_filename: str, name_pattern: str) -> ResNetFeatures:
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
    
class ProcessingSteps:
    """
    Provides static methods for data processing steps, including filtering,
    preprocessing (dimensionality reduction), clustering, and result evaluation.

    This class is not intended to be instantiated.
    """

    def __init__(self):
        ServiceFuncs.init_error(__class__.__name__)

    @staticmethod
    def filtration(features: ResNetFeatures) -> ResNetFeatures:
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
        ResNetFeatures.info_on_features(features)
        return features
    
    @staticmethod
    def run_preprocessing(features: ResNetFeatures, pipe_str: str ='PCA+UMAPND', 
                          params: Optional[Dict[str, dict]] =None) -> pd.DataFrame:
        """
        Applies a sequence of dimensionality reduction steps to the feature matrix.

        Parameters
        ----------
        features : ResNetFeatures
            The extracted and optionally filtered features.
        pipe_str : str, optional
            Pipeline specification string, e.g., 'PCA+UMAPND'.
        params : dict, optional
            Optional parameters for preprocessing steps.

        Returns
        -------
        pandas.DataFrame
            Transformed feature DataFrame with original metadata preserved.   
        """
        if params is None:
            params = {}

        print('Preprocessing')
        logger.info('Preprocessing')
        preprop = FeaturesPreprocessing(features)
        df = InputManager.check_if_database(features)
        processed = preprop.wrapper_preprop(df, pipe_str, params)
        return processed
    
    @staticmethod
    def run_clustering(features: ResNetFeatures, results_dir: str, model_type: str ='hdbscan', 
                       params: Optional[Dict[str, Any]]=None) -> Clustering:
        """
        Performs clustering and sorts image files into corresponding folders.

        Parameters
        ----------
        features : pandas.DataFrame
            The feature matrix with metadata.
        results_dir : str
            Path to output directory.
        model_type : str, optional
            Clustering model identifier.
        params : dict, optional
            Model parameters.

        Returns
        -------
        Clustering
            Clustering object with updated label assignments and organized folders.
        """
        print('Clusterization')
        logger.info(f'Clusterization by {model_type}')
        
        clusters = Clustering(features, results_dir)
        df_features, _ = DBFuncs.split_into_two(features)
        _, num_clusters = clusters.do_clustering(df_features, model_type=model_type, params=params)
        print(f'Number of clusters 20D: {num_clusters}')
        clusters.update_database()
        clusters.organize_files_by_cluster()
        return clusters
    
    @staticmethod
    def evaluation_clustering_results(clusters: Clustering, source_features: ResNetFeatures, model_type: str) -> None:
        """
        Display clustering quality scores and perform visualization.

        Parameters
        ----------
        clusters : Clustering
            Clustered object to evaluate.
        source_features : ResNetFeatures
            Original features to visaulize.
        model_type : str
            Clustering model identifier.
        """
        #Scores
        clusters.scores(clusters.df, clusters.labels)
        # Visualization
        logger.info('Visualization')
        clusters.visualize(source_features.database, model_cluster=model_type)
        plt.show()

    @staticmethod
    def evaluation_preprocessing_results(processed_database: ResNetFeatures, pipeline: str) -> None:
        """
        Show histogram of feature variances and preprocessing info.

        Parameters
        ----------
        processed_database : pd.DataFrame
            Preprocessed feature data.
        pipeline : str
            Pipeline descriptor used.
        """
        visualize = InputManager.get_bool('Would you like to plot a histogram of features varience? > ')
        ResNetFeatures.info_on_features(df=processed_database, visualize=visualize, title=pipeline)
        plt.show()
    
class ProcessingPipeline:
    """
    Orchestrates the full or step-wise processing of feature data.

    This class is not intended to be instantiated.
    """
    def __init__(self):
        ServiceFuncs.init_error(__class__.__name__)

    @staticmethod
    def run_processing(features: ResNetFeatures, default_filename: str, results_dir: str) -> None:
        """
        Offers the user a choice between standard pipeline execution or custom steps.

        Parameters
        ----------
        features : ResNetFeatures
            Feature database.
        default_filename : str
            Base name for saving intermediate or final results.
        results_dir: str
            Directory for result outputs.

        """
        message = 'Would you like to launch the standard processing algorithm (1) or \n' + \
                    'to call separate blocks of processing manually (2)?'
        print(message)
        logger.info('Processing running')
        while True:
            choice = InputManager.input_wrapper('Choose scenario [1/2] > ').strip()
            logger.info(f'Choice: {choice}')
            if choice == '1':
                ProcessingPipeline.standard_algorithm(features, default_filename, results_dir)
            elif choice == '2':
                ProcessingPipeline.choose_block(features, default_filename, results_dir)
            elif choice == 'break':
                break
            else:
                print('Invalid input. \n Try again! > ')
                logger.debug('Invalid input. Please enter 1 or 2 (or break).')



    @staticmethod
    def standard_algorithm(features: ResNetFeatures, default_filename: str, results_dir: str) -> None:
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
        results_dir: str
            Output directory.
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

        features = ProcessingSteps.filtration(features)

        message = f'Enter a file name to save the filtered data (or press "Enter" to use default name: {default_filename}) > '
        PathManager.saving_database(features.database, default_filename, message, suf='_filtered')

        processed = ProcessingSteps.run_preprocessing(features, 'PCA+UMAPND')

        clusters = ProcessingSteps.run_clustering(processed, results_dir)

        message = f'Enter a file name to save the clustered data (or press "Enter" to use default name: {default_filename}) > '
        PathManager.saving_database(clusters.df, default_filename, message, suf='_clustered')

        ProcessingSteps.evaluation_clustering_results(clusters, features, 'HDBSCAN')

    @staticmethod
    def choose_block(features: ResNetFeatures, default_filename: str, results_dir: str) -> None:

        """
        Provides UI to manually select individual processing steps.

        Parameters
        ----------
        features : ResNetFeatures
            Feature data.
        default_filename : str
            Output file base name.
        results_dir : str
            Directory for results.
        """

        message = 'Choose a block of data processing: \n' + \
        '1. preprocessing: scaling and dimensions reduction \n' + \
        '2. clusterization of processed data (compress dimensions first!) \n' + \
        'or enter "break" to return to the higher level'

        print(message)
        processed = None
        while True:
            choice = InputManager.input_wrapper('Choose a block [1/2] >  ').strip()
            logger.info(f'Choice: {choice}')
            if choice == '1':
                processed = PipelineUI.preprocessing_setup(features, default_filename)
            elif choice == '2':
                data = PipelineUI.source_data(processed)
                if data is None:
                    break
                else:
                    PipelineUI.clustering_setup(features, data, default_filename, results_dir)
            elif choice == 'break':
                break
            else:
                print('Invalid input. \n Please enter 1, 2 or 3 (or break). >')
                logger.debug('Invalid input.')

class PipelineUI:
    """
    Provides user interface utilities for preprocessing and clustering configuration and result handling.
    """
    def __init__(self):
        ServiceFuncs.init_error(__class__.__name__)

    @staticmethod
    def source_data(processed: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """
        Chooses whether to use previous step data or load from file.

        Parameters
        ----------
        processed : pd.DataFrame or None
            Data from previous pipeline step.

        Returns
        -------
        pd.DataFrame or None
            Selected feature data.
        """
        message = 'Would you like to use the results from the previous step? (1) \n' + \
        'or to set a path to a data file? (2)'
        print(message)
        choice = InputManager.input_wrapper('[1/2] > ').strip()
        logger.info(f'Choice: {choice}')
        if choice == '1':
            if processed is None:
                print('You should preprocess data first or specify a file path')
                logger.warning('Attempted clustering before preprocessing.')
        elif choice == '2':
            default_path = os.path.join(os.getcwd(), 'results', 'pearson_diagram_data_processed')
            path = PathManager.get_path('Set path to the source file > ', default_path)
            processed = DBFuncs.read_database(path)
        return processed
    
    @staticmethod
    def get_processing_params(pipeline: str) -> Dict[str, Dict[str, Any]]:
        """
        Prompts user to input parameters for each step in a pipeline.

        Parameters
        ----------
        pipeline : str
            Pipeline string.

        Returns
        -------
        dict
            Stepwise parameter dictionary.
        """
        print(f'Enter parameters for every or some of steps in {pipeline} >')
        models = pipeline.lower().split('+')
        params = {}
        for name in models:
            param = InputManager.input_dict(f'Enter parameters for {name} (as key=value) > ')
            if param:
                params[name] = param
        return params
    
    @staticmethod
    def get_clustering_params(model: str) -> Dict[str, Any]:
        """
        Prompts user for clustering parameters.

        Parameters
        ----------
        model : str
            Model name.

        Returns
        -------
        dict
            Clustering parameters.
        """
        print(f'Enter parameters model {model} >')

        params = InputManager.input_dict(f'Enter parameters for {model} (as key=value) > ')

        return params
    
    @staticmethod
    def clustering_setup(features: ResNetFeatures, data: pd.DataFrame, 
                         default_filename: str, results_dir: str) -> None:
        """
        Guides user through clustering setup and runs clustering.

        Parameters
        ----------
        features : ResNetFeatures
            Original features.
        data : pd.DataFrame
            Processed feature matrix.
        default_filename : str
            Output file base name.
        results_dir : str
            Directory for result output.
        """
        message = 'To cluster the data, enter a model name. \n' + \
        'Available models: \n' + \
        'hdbscan.HDBSCAN - use "hdbscan" \n' + \
        'sklearn.cluster.DBSCAN - use "dbscan" \n' + \
        'sklearn.cluster.KMeans - use "kmeans" \n' + \
        'You can specify parameters for any of these models (enter "params") \n' + \
        'or use default parameters: \n' + \
        'default parameters: \n' + \
        f'{Clustering(data, results_dir).default_params}.'
        print(message)

        while True:
            model = InputManager.input_wrapper('Enter a model name for clustering: ').strip()
            logger.info(f'Model: {model}')
            if model == 'break':
                break
            if InputManager.get_bool('Would you like to enter params? '):
                params = PipelineUI.get_clustering_params(model)
            else:
                params = {}
            clusters = ProcessingSteps.run_clustering(features=data, \
                                results_dir=results_dir, model_type=model, params=params)

            PipelineUI.to_deal_with_results_clustering(clusters, features, model, default_filename, '_clustering')

    @staticmethod
    def preprocessing_setup(features: ResNetFeatures, default_filename: str) -> Optional[pd.DataFrame]:
        """
        Guides user through pipeline setup and runs preprocessing.

        Parameters
        ----------
        features : ResNetFeatures
            Original feature data.
        default_filename : str
            File name base for saving.

        Returns
        -------
        pd.DataFrame
            Processed feature data.
        """
        message = 'To preprocess, enter a pipeline in the form \n' + \
        '"pca+umapnd+<...>". \n' + \
        'You can choose any (reasonable) combinations of \n' + \
        'sklearn.Normalizer() - use "normalizer" \n' + \
        'sklearn.StandartScalar() - use "scalar" \n' + \
        'sklearn.PCA() - use "pca" \n' + \
        'umap.UMAP() - use "umapnd" (it compreses to 20D default). \n' + \
        'You can specify parameters for any of these functions (enter "params") \n' + \
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
                params = PipelineUI.get_processing_params(pipeline)
            else:
                params = {}
            processed = ProcessingSteps.run_preprocessing(features, pipeline, params)

            PipelineUI.to_deal_with_results_features(pipeline, processed, default_filename, '_processed')
        return processed
    
    @staticmethod
    def to_deal_with_results_features(pipeline: str, processed: pd.DataFrame, default_filename: str, suf: str) -> None:
        """
        Provides user options for processed feature output: save, visualize.

        Parameters
        ----------
        pipeline : str
            Pipeline string.
        processed : pd.DataFrame
            Resulting data.
        default_filename : str
            Output file base.
        suf : str
            Suffix to append to filename.
        """

        while True:
            choice = InputManager.input_wrapper('Enter "save", "print", "break" > ').strip()
            logger.info(f'Choice: {choice}')
            if choice == 'save':
                df = InputManager.check_if_database(processed)
                message = f'Enter a file name to save the data (or press "Enter" to use default_name: {default_filename}) > '
                PathManager.saving_database(df, default_filename, message, suf=suf)
            elif choice == 'print':
                ProcessingSteps.evaluation_preprocessing_results(processed, pipeline)
            elif choice == 'break':
                break
            else:
                print('Invalid input. \n Try again! > ')
                logger.debug('Invalid input.')


    @staticmethod
    def to_deal_with_results_clustering(clusters: 'Clustering', source_features: 'ResNetFeatures', model_type: str, default_filename: str, suf: str) -> None:
        """
        Provides user options for clustering result output.

        Parameters
        ----------
        clusters : Clustering
            Resulting clustering.
        source_features : ResNetFeatures
            Original features.
        model_type : str
            Model used.
        default_filename : str
            Output base name.
        suf : str
            Suffix for filename.
        """
        while True:
            choice = InputManager.input_wrapper('Enter "save", "print", "break" > ').strip()
            logger.info(f'Choice: {choice}')
            if choice == 'save':
                df = InputManager.check_if_database(clusters)
                message = f'Enter a file name to save the data (or press "Enter" to use default_name: {default_filename}) > '
                PathManager.saving_database(df, default_filename, message, suf=suf)
            elif choice == 'print':
                ProcessingSteps.evaluation_clustering_results(clusters, source_features, model_type)
            elif choice == 'break':
                break
            else:
                print('Invalid input. \n Try again! > ')
                logger.debug('Invalid input.')

