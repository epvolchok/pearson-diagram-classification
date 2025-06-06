#Copyright (c) 2025 Evgeniia VOLCHOK
#for contacts e.p.volchok@gmail.com

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0


import os
import shutil
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np

import hdbscan
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

from .libservice import ServiceFuncs, DBFuncs
from .libpreprocessing import FeaturesPreprocessing

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('qtagg')
from matplotlib import rc

rc('text', usetex=True)
rc('font', family='serif')
rc('text.latex', preamble=r"\usepackage[utf8]{inputenc}")
matplotlib.rcParams.update({'font.size': 14})

import logging
logger = logging.getLogger(__name__)

class Clustering:

    """
    Performs HDBSCAN clustering on a feature dataset and manages file sorting and visualization.

    This class supports unsupervised clustering using the HDBSCAN algorithm,
    updates the associated feature database with labels, and organizes files
    into directories based on cluster assignments. It also includes tools for
    cluster visualization using PCA and UMAP.

    Parameters
    ----------
    df : pandas.DataFrame
        Input feature DataFrame containing at least 'oldpath' and feature columns.
    copy : bool, optional
        If True, the input DataFrame is copied internally. Default is False.
    clear : bool, optional
        If True, the target directory './processed' will be cleared before use. Default is True.

    Attributes
    ----------
    df : pandas.DataFrame
        The working DataFrame, either copied or referenced.
    num_clusters : int
        Number of clusters found (excluding noise).
    labels : list of int
        Cluster labels assigned by HDBSCAN (-1 indicates noise).
    dir : str
        Directory path where clustered files will be organized (default is './processed').
    """

    def __init__(self, df: pd.DataFrame, results_dir: str, copy: bool = False, clear: bool = True) -> None:

        self.cluster_algorithms = {
            'HDBSCAN': hdbscan.HDBSCAN,
            'DBSCAN': DBSCAN,
            'KMeans': KMeans
        }

        self.default_params = {
            'hdbscan': {'type': 'HDBSCAN', 'params': {'min_cluster_size': 15, 'min_samples': 5, 'metric': 'euclidean'}},
            'kmeans': {'type': 'KMeans', 'params': {'n_clusters': 5, 'random_state': 42}},
            'dbscan': {'type': 'DBSCAN', 'params': {'eps': 0.5, 'min_samples': 5, 'metric': 'euclidean'}}
        }

        if copy:
            self.df = df.copy()
        else:
            self.df = df

        self.num_clusters = 0
        self.labels = None

        self.dir = results_dir #os.path.join(os.getcwd(), 'processed')
        ServiceFuncs.preparing_folder(self.dir, clear=clear)

    def _init_model(self, model_type: str, params: dict)-> Any:
        """
        Initialize a clustering model with the given type and parameters.

        Parameters
        ----------
        model_type : str
            Name of the clustering algorithm.
        params : dict
            Parameters to initialize the clustering model.

        Returns
        -------
        Any
            Instantiated clustering model.

        Raises
        ------
        ValueError
            If the model type is unknown.
        """
        if model_type not in self.cluster_algorithms:
            logger.error(ValueError(f'Unknown model: {model_type}'))
            raise ValueError(f'Unknown model: {model_type}')
        logger.info(f'Initialize model {model_type}')
        return self.cluster_algorithms[model_type](**params)
    
    def setup_clustering(self, model: str, params: Optional[Dict[str, dict]] = None) -> Tuple[str, object]:
        """
        Sets up the clustering model using default or custom parameters.

        Parameters
        ----------
        model : str
            Name of the clustering algorithm.
        params : dict, optional
            Custom parameters for the clustering algorithm.

        Returns
        -------
        tuple
            (model_type, model_instance)

        Raises
        ------
        ValueError
            If the model configuration is not found.
        """
        if params is None:
            params = {}
        model = model.lower()
        if model not in self.default_params:
            logger.error(ValueError(f'Model config for "{model}" not found'))
            raise ValueError(f'Model config for "{model}" not found')
        
        if model not in params:
                model_cfg = self.default_params[model]['params']
        else:
            model_cfg = params[model]
        model_type = self.default_params[model]['type']
        model_clustering = self._init_model(model_type, model_cfg)
        logger.info(f'Clustering model: {model_type}, params: {model_cfg}')
        return model_type, model_clustering

    def do_clustering(self, df: pd.DataFrame, model_type: str, params: Optional[Dict[str, dict]] = None) -> Tuple[np.ndarray, int]:
        """
        Applies clustering to the given feature matrix.

        Parameters
        ----------
        df : pandas.DataFrame or numpy.ndarray
            Feature matrix.
        model_type : str
            Name of the clustering algorithm.
        params : dict, optional
            Parameters for the clustering algorithm.

        Returns
        -------
        tuple
            (labels, number_of_clusters)
        """
        if params is None:
            params = {}
        model, model_clustering = self.setup_clustering(model_type, params)
        logger.info(f'Clusterization with {model}')
        
        
        if model == 'HDBSCAN':
            self.labels = model_clustering.fit_predict(df)
        else:
            cluster = model_clustering.fit(df)
            self.labels = cluster.labels_

        self.num_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        logger.info(f'Number of clusters: {self.num_clusters}')
        return self.labels, self.num_clusters
    
    def update_database(self) -> None:
        """
        Inserts the cluster labels into the internal DataFrame as a new column 'label'.
        """
        try:
            self.df.insert(1, 'label', self.labels)
            logger.debug('Insertion of cluster labels to the database.')
        except Exception as err:
            logger.error(f'Labels can not be added to the database: {err}')

    def create_dirs(self, clear: str =False) -> None:
        """
        Creates output directories for each cluster and a separate one for noise.

        Parameters
        ----------
        clear : bool
            Whether to clear the directory before creation.
        """
        for cl in range(self.num_clusters):
            dir_name = os.path.join(self.dir, 'label_'+str(cl))
            ServiceFuncs.preparing_folder(dir_name, clear=clear)

        noise_name = os.path.join(self.dir,'noise')
        ServiceFuncs.preparing_folder(noise_name, clear=clear)
        logger.debug('Folders for sorted data created')



    def create_newpath(self) -> None:
        """
        Assigns new output paths for each sample in the DataFrame based on their cluster label.
        Clustered samples go to './processed/label_<cluster>', noise samples go to './processed/noise'.
        """
         
        for cl in range(self.num_clusters):
            self.df.loc[self.df['label'] == cl, 'path'] = os.path.join(self.dir, 'label_'+str(cl))
        self.df.loc[self.df['label'] == -1, 'path'] = os.path.join(self.dir, 'noise')
        logger.debug('New paths added')


    def copy_files(self) -> None:
        """
        Copies image files from their original locations ('oldpath') to the new cluster-assigned directories ('path').

        Skips rows where either path is missing or not a string. Errors are printed but do not interrupt execution.
        """

        for index, row in self.df.iterrows():
            oldpath = row['oldpath']
            newpath = row['path']
            if isinstance(oldpath, str) and isinstance(newpath, str):
                try:
                    shutil.copy2(oldpath, newpath)
                    
                except OSError as e:
                    print(f'Error during copying {oldpath} -> {newpath}: {e}')
                    logger.warning(f'Error during copying {oldpath} -> {newpath}: {e}')
            else:
                print(f'Probably missing path {index}: oldpath={oldpath}, path={newpath}')
                logger.warning(f'Probably missing path {index}: oldpath={oldpath}, path={newpath}')
        logger.debug('Files copied to the new folders')
        
    def organize_files_by_cluster(self) -> None:
        """
        Organizes image files into directories by cluster.

        This method:
        - Creates cluster directories
        - Assigns new file paths
        - Copies files to new locations
        - Removes the 'oldpath' column from the DataFrame
        
        Returns
        -------
        pandas.DataFrame 
            resulted database self.df
        """

        self.create_dirs()
        self.create_newpath()
        self.copy_files()
        self.df.drop('oldpath', axis=1, inplace=True)

        return self.df


    def visualize(self, df: pd.DataFrame, model_cluster: str = 'HDBSCAN', 
            model_preprop: str = 'PCA+UMAP2D', params: Optional[Dict[str, dict]] = None, filename: Optional[str] = None) -> None:
        """
        Reduces the feature space to 2D using a preprocessing pipeline,
        applies clustering, and plots the results.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame containing features.
        model_cluster : str
            Clustering algorithm to use.
        model_preprop : str
            Preprocessing pipeline string (e.g., 'PCA+UMAP2D').
        params : dict, optional
            Dictionary of parameters for the clustering algorithm.
        filename : str, optional
            Output filename (without extension) for saving the plot.
        """
        if params is None:
            params = {}
        features_processed, labels, _ = self.visdata_preparation(df, model_cluster, model_preprop, params)

        plt.figure(figsize=(10, 8))
        palette = plt.get_cmap('tab10')

        for label in set(labels):
            mask = labels == label
            color = 'gray' if label == -1 else palette(label % 10)
            plt.scatter(features_processed[mask, 0], features_processed[mask, 1], s=10, color=color, label=f'Cluster {label}' if label != -1 else 'Noise')

        plt.legend()
        plt.title(f'{model_preprop} + {model_cluster}: clustering visualization')
        plt.xlabel('UMAP-1 component')
        plt.ylabel('UMAP-2  component')

        self.saving_figs(filename)



    def visdata_preparation(self, df: pd.DataFrame, model_cluster: str = 'HDBSCAN', 
            model_preprop: str = 'PCA+UMAP2D', params: Optional[Dict[str, dict]] = None) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Prepares data for visualization: preprocesses features and applies clustering.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame.
        model_cluster : str
            Clustering algorithm.
        model_preprop : str
            Preprocessing pipeline.
        params : dict, optional
            Clustering parameters.

        Returns
        -------
        tuple
            (2D_features, labels, number_of_clusters)
        """
        if params is None:
            params = {}
        df_features, _= DBFuncs.split_into_two(df)
        features_processed = FeaturesPreprocessing(df, copy=True).preprocessing(df_features, model_preprop)
        labels, n_clusters = self.do_clustering(features_processed, model_cluster, params)
        print(f'Found {n_clusters} clusters (2D)')
        logger.info(f'Visualization. Found {n_clusters} clusters (2D)')
        return features_processed, labels, n_clusters

    def saving_figs(self, filename: Optional[str]) -> None:
        """
        Saves the clustering plot in PDF and PNG formats.

        Parameters
        ----------
        filename : str
            Base filename to use for saving.
        """
        fig_dir = os.path.join(os.getcwd(), 'figures')
        default_name = 'clusters'
        if not filename:
            filename = default_name
        file_png = os.path.join(fig_dir, filename+'.png')
        file_pdf = os.path.join(fig_dir, filename+'.pdf')
        plt.savefig(file_pdf, format='pdf')
        plt.savefig(file_png, format='png', dpi=300)

    def scores(self, df: pd.DataFrame, labels: np.ndarray) -> None:
        """
        Computes and prints clustering evaluation scores.

        Parameters
        ----------
        df : pandas.DataFrame
            Full DataFrame including features.
        labels : numpy.ndarray
            Cluster labels for each sample.

        Prints
        ------
        Silhouette Score and Davies-Bouldin Index.
        """
        excluded_columns=['dataset_name', 'date', 'dist_to_sun[au]', 'SAMPLES_NUMBER', 
                          'SAMPLING_RATE[kHz]', 'SAMPLE_LENGTH[ms]', 'path']
        df_features, _= DBFuncs.split_into_two(df, excluded_columns)
        sil_score = silhouette_score(df_features, labels)
        print(f'Silhouette Score: {sil_score:.2f}')
        logger.info(f'Silhouette Score: {sil_score:.2f}')

        dbi = davies_bouldin_score(df_features, labels)
        print(f'Davies-Bouldin index: {dbi:.2f}')
        logger.info(f'Davies-Bouldin index: {dbi:.2f}')

    def stability_score(self, method1: str, labels1: np.array, method2: str, labels2: np.array) -> None:
        """
        Computes and prints Adjusted Rand Index for labels of two clustering methods.

        Parameters
        ----------
        methods1, methods2 : str
            Names of used clustering methods
        labels1, labels2 : np.array
            Arrays with corresponding labels of clusters

        Prints
        ----------
        Adjusted Rand Index
        """
        logger.info(f'Comparison of {method1} and {method2} clustering methods')
        print(f'Comparison of {method1} and {method2} clustering methods')

        ARS = adjusted_rand_score(labels1, labels2)

        print(f'Adjusted Rand Index: {ARS:.2f}')
        logger.info(f'Adjusted Index score: {ARS:.2f}')