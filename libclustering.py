#Copyright (c) 2025 Evgeniia VOLCHOK
#for contacts e.p.volchok@gmail.com

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import pandas as pd
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
        clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5, metric='euclidean')
        self.labels = clusterer.fit_predict(df)
        self.num_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        return self.labels, self.num_clusters
    
    def update_database(self):
        self.df.insert(1, 'label', self.labels)
        self.df.head()

    def create_dirs(self):
        if os.path.exists(self.dir):
            for cl in range(self.num_clusters):
                dir_name = self.dir+'/label_'+str(cl)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
            if not os.path.exists(self.dir+'/noise'):
                os.makedirs(self.dir+'/noise')


    def create_newpath(self):
        for cl in range(self.num_clusters):
            self.df.loc[self.df['label'] == cl, 'path'] = './processed/label_'+str(cl)
        self.df.loc[self.df['label'] == -1, 'path'] = './processed/noise'


    def copy_files(self):
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
        self.create_dirs()
        self.create_newpath()
        self.copy_files()
        self.df.drop('oldpath', axis=1, inplace=True)


    def visualize_HDBSCAN(self, df):
        excluded_columns = ['dataset_name', 'date', 'dist_to_sun[au]', 'SAMPLES_NUMBER', 'SAMPLING_RATE[kHz]', \
                    'SAMPLE_LENGTH[ms]', 'path']
        df_features, excluded_part = ServiceFuncs.split_into_two(df)
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

