#Copyright (c) 2025 Evgeniia VOLCHOK
#for contacts e.p.volchok@gmail.com

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import Pipeline
import umap
import pandas as pd
from libservice import ServiceFuncs, DBFuncs

import logging
logger = logging.getLogger(__name__)

class FeaturesPreprocessing:

    
    """
        Applies dimensionality reduction and scaling pipelines to a DataFrame of features.

        This class provides a flexible wrapper around common preprocessing techniques 
        such as PCA, StandardScaler, and UMAP. The sequence of transformations is defined 
        by a string-based pipeline syntax.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame containing both metadata and features.
        copy : bool, optional
            If True, the input DataFrame is copied before processing.
            If False (default), the original DataFrame is modified in-place.

        Attributes
        ----------
        models : dict
            Dictionary mapping method names ('pca', 'scaler', 'umapnd', 'umap2d') to
            their corresponding scikit-learn or UMAP objects.
        df : pandas.DataFrame
            Stored copy or reference to the input DataFrame.
    """

    def __init__(self, df, config=None, copy=False):

        self.df = df.copy() if copy else df

        self.model_registry = {
            'Normalizer': Normalizer,
            'StandardScaler': StandardScaler,
            'PCA': PCA,
            'UMAP': umap.UMAP,
            }
        
        self.default_params = {
            'normalizer': {'type': 'Normalizer', 'params': {}},
            'standardscaler': {'type': 'StandardScaler', 'params': {}},
            'scaler': {'type': 'StandardScaler', 'params': {}},
            'pca': {'type': 'PCA', 'params': {'n_components': 0.95, 'svd_solver': 'full'}},
            'umapnd': {'type': 'UMAP', 'params': {'n_components': 20, 'min_dist': 0.1, 'metric': 'cosine'}},
            'umap2d': {'type': 'UMAP', 'params': {'n_components': 2, 'min_dist': 0.1, 'metric': 'cosine'}}
        }


    def _init_model(self, model_type: str, params: dict):
        if model_type not in self.model_registry:
            logger.error(ValueError(f'Unknown model: {model_type}'))
            raise ValueError(f'Unknown model: {model_type}')
        logger.info(f'Initialize model {model_type}')
        return self.model_registry[model_type](**params)
    

    def build_pipeline(self, pipe_str: str, params: dict ={}):
        models = pipe_str.lower().split('+')
        steps = []
        for name in models:
            if name not in self.default_params:
                logger.error(ValueError(f'Model config for "{name}" not found'))
                raise ValueError(f'Model config for "{name}" not found')
            if name not in params:
                model_cfg = self.default_params[name]['params']
            else:
                model_cfg = params[name]
            model_type = self.default_params[name]['type']
            model = self._init_model(model_type, model_cfg)
            logger.info(f'Model: {model_type}, params: {model_cfg}')
            steps.append((name, model))
        return Pipeline(steps)

    def preproccessing(self, df, pipe_str, params={}): #pipe_str='PCA+UMAPnd+..'
        """
        Applies a sequence of preprocessing transformations to the feature data.

        Parameters
        ----------
        df : pandas.DataFrame
            Feature matrix (excluding metadata).
        pipe_str : str
            A string representing the sequence of transformations to apply,
            joined by '+'. For example: 'pca+umap2d'.

        Returns
        -------
        numpy.ndarray or None
            Transformed feature array if input is non-empty; otherwise None.

        Raises
        ------
        KeyError
            If the pipeline string contains unknown model names.
        """
        logger.info(f'Preprocessing launched with "{pipe_str}" pipeline')
        if df.size == 0:
            return None
        pipe = self.build_pipeline(pipe_str, params)
        return pipe.fit_transform(df)
        
    def wrapper_preprop(self, df, pipe_str, params={}):

        """
        Applies preprocessing to the feature part of a DataFrame while preserving metadata.

        This method splits the input DataFrame into metadata and feature parts,
        applies the specified transformation pipeline to the features, and then
        merges the result back with the metadata.

        Parameters
        ----------
        df : pandas.DataFrame
            Full input DataFrame including metadata and features.
        pipe_str : str
            Pipeline string specifying preprocessing steps (e.g., 'scaler+pca+umap2d').

        Returns
        -------
        pandas.DataFrame
            New DataFrame containing preserved metadata and transformed features.

        Raises
        ------
        ValueError
            If feature transformation fails or input DataFrame is empty.
        """

        df_features, excluded_part = DBFuncs.split_into_two(df)
        processed = self.preproccessing(df_features, pipe_str, params)
        df_processed = pd.DataFrame(
            processed,
            columns=[f'feat_{i}' for i in range(processed.shape[1])]
        )
        df_new = pd.concat([excluded_part, df_processed], axis=1)
        return df_new