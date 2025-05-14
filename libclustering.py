#Copyright (c) 2025 Evgeniia VOLCHOK
#for contacts e.p.volchok@gmail.com

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0


import os
import subprocess

import hdbscan

from libservice import ServiceFuncs
from libpreprocessing import FeaturesPreprocessing
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('qtagg')
from matplotlib import rc

rc('text', usetex=True)
rc('font', family='serif')
rc('text.latex', preamble=r"\usepackage[utf8]{inputenc}")
matplotlib.rcParams.update({'font.size': 14})


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

    def __init__(self, df, copy=False, clear=True):

        if copy:
            self.df = df.copy()
        else:
            self.df = df
        self.num_clusters = 0
        self.labels = []
        self.dir = './processed'
        ServiceFuncs.preparing_folder(self.dir, clear=clear)


    def clustering_HDBSCAN(self, df):
        """
        Applies HDBSCAN clustering to the given feature matrix.

        Parameters
        ----------
        df : pandas.DataFrame or numpy.ndarray
            Feature matrix with shape (n_samples, n_features).

        Returns
        -------
        labels : numpy.ndarray
            Array of cluster labels for each sample (-1 indicates noise).
        num_clusters : int
            Number of clusters detected (excluding noise).
        """
        
        clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5, metric='euclidean')
        self.labels = clusterer.fit_predict(df)
        self.num_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        return self.labels, self.num_clusters
    
    def update_database(self):
        """
        Inserts the cluster labels into the internal DataFrame as a new column 'label'.
        """

        self.df.insert(1, 'label', self.labels)
        self.df.head()

    def create_dirs(self):
        """
        Creates output directories for each cluster and a separate one for noise.
        The directories are created under `self.dir` if they do not already exist.
        """
        if os.path.exists(self.dir):
            for cl in range(self.num_clusters):
                dir_name = self.dir+'/label_'+str(cl)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
            if not os.path.exists(self.dir+'/noise'):
                os.makedirs(self.dir+'/noise')


    def create_newpath(self):
        """
        Assigns new output paths for each sample in the DataFrame based on their cluster label.
        Clustered samples go to './processed/label_<cluster>', noise samples go to './processed/noise'.
        """
         
        for cl in range(self.num_clusters):
            self.df.loc[self.df['label'] == cl, 'path'] = './processed/label_'+str(cl)
        self.df.loc[self.df['label'] == -1, 'path'] = './processed/noise'


    def copy_files(self):
        """
        Copies image files from their original locations ('oldpath') to the new cluster-assigned directories ('path').

        Skips rows where either path is missing or not a string. Errors are printed but do not interrupt execution.
        """

        for index, row in self.df.iterrows():
            oldpath = row['oldpath']
            newpath = row['path']
            if isinstance(oldpath, str) and isinstance(newpath, str):
                try:
                    subprocess.run(['cp', '-u', row['oldpath'], row['path']], check=True)
                except subprocess.CalledProcessError as e:
                    print(f'Error during copying {row['oldpath']} → {row['path']}: {e}')
            else:
                print(f'Probably missing path {index}: oldpath={oldpath}, path={newpath}')
        
    def sort_files(self):
        """
        Organizes image files into directories by cluster.

        This method:
        - Creates cluster directories
        - Assigns new file paths
        - Copies files to new locations
        - Removes the 'oldpath' column from the DataFrame
        """

        self.create_dirs()
        self.create_newpath()
        self.copy_files()
        self.df.drop('oldpath', axis=1, inplace=True)


    def visualize_HDBSCAN(self, df):
        """
        Reduces the feature space to 2D using PCA + UMAP, applies HDBSCAN clustering, and plots the results.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame containing features and metadata.

        Saves
        -----
        './figures/clusterization_triggered.pdf' : PDF version of the cluster plot
        './figures/clusterization_triggered.png' : PNG version (300 DPI)

        Shows
        -----
        A scatter plot of the clustered data with noise in gray and clusters in color.
        """
        
        df_features, _= ServiceFuncs.split_into_two(df)
        features_processed = FeaturesPreprocessing(df, copy=True).preproccessing(df_features, 'PCA+UMAP2D')
        labels, n_clusters = self.clustering_HDBSCAN(features_processed)
        print(f'Found {n_clusters} clusters (2D)')
        plt.figure(figsize=(10, 8))
        palette = plt.get_cmap('tab10')

        for label in set(labels):
            mask = labels == label
            color = 'gray' if label == -1 else palette(label % 10)
            plt.scatter(features_processed[mask, 0], features_processed[mask, 1], s=10, color=color, label=f'Cluster {label}' if label != -1 else 'Noise')

        plt.legend()
        plt.title('PCA + UMAP2D + HDBSCAN: clustering visualization')
        plt.xlabel('UMAP-1 component')
        plt.ylabel('UMAP-2  component')
        
        if not os.path.exists('./figures'):
            os.makedirs('./figures')

        plt.savefig('./figures/clusterization_triggered.pdf', format='pdf')
        plt.savefig('./figures/clusterization_triggered.png', format='png', dpi=300)

        plt.show()

