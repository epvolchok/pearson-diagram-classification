#Copyright (c) 2025 Evgeniia VOLCHOK
#for contacts e.p.volchok@gmail.com

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0


import pandas as pd
import numpy as np
import os
import re
from PIL import Image

from torchvision import models, transforms
import torch
from tqdm import tqdm
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

from functools import cached_property
from dataclasses import dataclass, field

from libservice import ServiceFuncs, DBFuncs

import logging
logger = logging.getLogger(__name__)

@dataclass
class ResNetFeatures:

    """
    Extracts deep features from image datasets using torchvision model ResNet50,
    merges them with metadata, and provides filtering and analysis tools.

    Attributes
    ----------
    path : str
        Path to the folder containing image files.
    info_path : str
        Path to the metadata text file with observation parameters.
    flag : str
        Mode of operation: 'read' to load precomputed database, 'extract' to generate it from images.
    device : str
        Computation device to use ('cuda' or 'cpu').
    filter_mixed : bool
        Whether to filter out mixed-frequency datasets.
    name_pattern : str
        Regex pattern to extract dataset names from image filenames.
    extra_params : dict
        Additional parameters for database loading/saving.
    """

    path: str
    info_path: str = os.path.join(os.getcwd(), 'data', 'SOLO_info_rswf.txt')
    flag: str = 'read'
    device: str = 'cuda'
    filter_mixed: bool = True
    name_pattern: str = r'(solo_L2_rpw-tds-surv-(?:r|t)swf-e_\d+\w+)'
    extra_params: dict = field(default_factory=dict)

    # inner filed, will be initialized later
    _device: torch.device = field(init=False)
    names: list = field(init=False)
    img_path: list = field(init=False)
    info: dict = field(init=False)
    database: object = field(init=False)
    
    def __post_init__(self):
        """
        Initializes internal fields, filters images, and loads or creates the feature database.
        """

        self.set_device()

        self.names = [f for f in os.listdir(self.path) if ServiceFuncs.check_extension(f)]
        self.img_path = [os.path.join(self.path, name) for name in self.names]

        self.info = DBFuncs.load_info(self.info_path)
        logger.debug(f'Path to the metadata file: {self.info}')

        if self.filter_mixed:
            self.filtering_imgs(self.path, self.name_pattern)
        logger.debug(f'Filtration: {self.filter_mixed}')

        logger.debug(f'Features load regime: {self.flag}')
        if self.flag == 'read':
            self.database = DBFuncs.read_database(**self.extra_params)
        elif self.flag == 'extract':
            self.database = self.create_database()
            DBFuncs.save_database(self.database, **self.extra_params)
        else:
            logger.error(f'Unknown flag: {self.flag}')
            raise ValueError(f'Unknown flag: {self.flag}')
        
    def set_device(self):
        """
        Sets torch.device according to a user wish self.device if it is available
        or CPU if it is not.
        """
        if self.device == 'cuda' and not torch.cuda.is_available():
            print('CUDA requested but not available. Falling back to CPU.')
            self._device = torch.device('cpu')
            logger.debug('CUDA requested but not available. Falling back to CPU.')
        else:
            self._device = torch.device(self.device)
        print(f'Using device: {self._device}')
        logger.info(f'Using device: {self._device}')

    @cached_property
    def model(self):
        """
        Loads and caches the model and input transformation.
        
        Returns
        -------
        tuple
            A tuple (model, transform) where model is a torch.nn.Module and transform is a torchvision transform pipeline.
        """
        logger.info('ResNet50 model init')

        transform = transforms.Compose([
        transforms.Resize((224)),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])
                                    ])

        model_resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model_resnet50 = torch.nn.Sequential(*list(model_resnet50.children())[:-1])  # without the last layer
        model_resnet50.to(self._device)

        return model_resnet50, transform

    def find_mixed_freq(self):
        """
        Identifies dataset names with mixed sampling frequencies in the metadata.

        Returns
        -------
        pandas.Series
            Series of dataset names to exclude.
        """
        high_freq = 524
        low_freq = 262

        tgt = self.info['SAMPLING_RATE[kHz]']
        mask = ~(
            ((tgt >= low_freq - 5) & (tgt <= low_freq + 5)) |
            ((tgt >= high_freq - 5) & (tgt <= high_freq + 5))
        )
        
        df = self.info[mask]['dataset_name']
        return df
    

    def filtering_imgs(self, path: str, name_pattern: str):
        """
        Filters out image paths (in self.img_path,self.names) corresponding to mixed-frequency datasets.

        Parameters
        ----------
        path : str
            Directory containing images.
        name_pattern : str
            Regular expression to extract dataset names from filenames.
        """
        logger.debug('Images filtration')
        mixed_freq = self.find_mixed_freq()
        
        for p in self.img_path:
            dataset_name = re.search(name_pattern, p).group(0)
            if dataset_name in mixed_freq.values:
                self.names.remove(dataset_name+'.png')
                filepath = os.path.join(path, dataset_name+'.png')
                self.img_path.remove(filepath)

    def features(self):
        """
        Extracts deep features from all images using ResNet50 model.

        Returns
        -------
        np.ndarray
            Array of feature vectors for all valid images.
        """
        logger.info('Feature extraction')
        model, transform = self.model
        model.eval()
        features = []
        for path in tqdm(self.img_path, desc='Extracting features'):
            try:
                with Image.open(path).convert('RGB') as img:
                    img_tensor = transform(img).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    feat = model(img_tensor).squeeze().cpu().numpy()
                features.append(feat)
            except OSError as err:
                logging.warning(f'Error while openning an image {path}: {err}')
                print(f'Error while openning an image {path}: {err}')
        return np.array(features)
    
    def features_to_db(self):
        features = self.features()
        try:
            df_features = pd.DataFrame(
                features,
                columns=[f'feat_{i}' for i in range(features.shape[1])]
            )
            self.names = [name[:-4] for name in self.names]
            df_features.insert(0, 'oldpath', self.img_path)
            df_features.insert(0, 'dataset_name', self.names)
            logger.debug('Numpy features successfully saved in a database.')
            return df_features
        except Exception as err:
            logging.error(f'Error during creation of the database from features: {err}')
            raise ValueError
        
    
    def create_database(self):
        """
        Extracts features, merges them with metadata, and constructs a full DataFrame.

        Returns
        -------
        pandas.DataFrame
            Feature database with metadata.
        """
        df_features = self.features_to_db()
        
        df_info = self.info[['dataset_name', 'date', 'dist_to_sun[au]', 'SAMPLES_NUMBER', 'SAMPLING_RATE[kHz]', 'SAMPLE_LENGTH[ms]']]
            
        df_full = pd.merge(
                df_info, 
                df_features, 
                how='left', on='dataset_name')
        df_full.dropna(inplace=True, ignore_index=True)
        print('Database with information about observation parameters and extracted features:')
        logger.info('Database on features and observation parameters description created')
        print(df_full.info())
        print(df_full.head())
        return df_full


    def filtering_nonzerocolumns(self):
        """
        Removes columns in the feature matrix that contain only zeros.

        Returns
        -------
        pandas.DataFrame
            Filtered DataFrame with non-zero feature columns.
        """
        logger.info('Filtarion of columns in the feature matrix that contain only zeros')

        df_features, excluded_part = DBFuncs.split_into_two(self.database)
        non_zero_columns = ~(df_features == 0).all(axis=0)
        filtered_features = df_features.loc[:, non_zero_columns]
        final_df = pd.concat([excluded_part, filtered_features], axis=1)

        print('Filtration of zero columns. Remaining size:')
        logger.info(f'Remaining size: {final_df.shape[1]}')
        print(final_df.shape[1])
        return final_df
    
    def filtering_by_variance(self, threshold=5e-5):
        """
        Removes low-variance features from the database.

        Parameters
        ----------
        threshold : float
            Minimum variance required for a feature to be retained.

        Returns
        -------
        pandas.DataFrame
            Filtered DataFrame with high-variance features.
        """
        logger.info('Filtarion of low-variance features')

        df_features, excluded_part = DBFuncs.split_into_two(self.database)
        selector = VarianceThreshold(threshold=threshold)
        filtered_features = selector.fit_transform(df_features)
        df_filtered = pd.DataFrame(
            filtered_features,
            columns=[f'feat_{i}' for i in range(filtered_features.shape[1])]
        )

        print('Filtration by the variance threshold. Remaining size:')
        final_df = pd.concat([excluded_part, df_filtered], axis=1)
        logger.info(f'Remaining size: {final_df.shape[1]}')
        print(final_df.shape[1])
        return final_df
    
    def info_on_features(self, visualize=False, title=''):
        """
        Prints statistics on extracted features and optionally visualizes variance distribution.

        Parameters
        ----------
        visualize : bool
            If True, show a histogram of feature variances.
        title : str
            Optional title for the plot.
        """
        df_features, _ = DBFuncs.split_into_two(self.database)

        distances = pairwise_distances(df_features, metric='cosine')
        mean_dist = distances[np.triu_indices_from(distances, k=1)].mean()

        print(f'Average cosine distance between embeddings: {mean_dist}')
        logger.info(f'Average cosine distance between embeddings: {mean_dist}')
        variances = np.var(df_features, axis=0)
        print(f'Average variance: {np.mean(variances)}')
        logger.info(f'Average variance: {np.mean(variances)}')

        if visualize:
            self.visualize_variance(variances, title=title)

    def visualize_variance(self, variances, title=''):
        plt.figure(figsize=(7,5))
        plt.subplot(1, 2, 1)
        plt.hist(variances, bins=50, color='skyblue')
        plt.ylabel('Frequency')
        if title:
            plt.title(title)
        else:
            plt.title('Variance')


