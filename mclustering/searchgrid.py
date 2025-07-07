from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from hdbscan import HDBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone
from joblib import Parallel, delayed
import json
import pickle
import os
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('qtagg')
from matplotlib import rc
from matplotlib import gridspec

rc('text', usetex=True)
rc('font', family='serif')
rc('text.latex', preamble=r"\usepackage[utf8]{inputenc}")
matplotlib.rcParams.update({'font.size': 14})

import pandas as pd
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class GridSearch:
    """
    Performs unsupervised model selection for clustering pipelines using UMAP 
    for dimensionality reduction and various clustering algorithms.

    Parameters
    ----------
    df : pandas.DataFrame
        Feature matrix to use in grid search.
    copy : bool, optional
        If True, creates a copy of the input DataFrame.
    """

    def __init__(self, df: pd.DataFrame, copy: bool = False) -> None:
        self.df = df.copy() if copy else df

        self.param_umap = {
        'umap__n_components': [10, 15, 20],
        'umap__n_neighbors': [2, 15, 30]
        }

        self.models = {
            'kmeans': KMeans(random_state=42),
            'dbscan': DBSCAN(),
            'hdbscan': HDBSCAN(prediction_data=True)
        }

        self.param_models = {
            'kmeans': {'kmeans__n_clusters': [4, 5, 6], 'kmeans__algorithm': ['lloyd', 'elkan']},
            'dbscan': {'dbscan__eps': [0.15, 0.35, 0.5, 0.65, 0.85]},
            'hdbscan': {'hdbscan__min_cluster_size': [5, 10, 15], 
                        'hdbscan__cluster_selection_epsilon': [0.15, 0.35, 0.5, 0.65, 0.85]}
        }


    @staticmethod
    def evaluate_pca(df: pd.DataFrame) -> np.ndarray:
        """
        Applies PCA to the dataset with 95% variance retained.

        Parameters
        ----------
        df : pandas.DataFrame
            Input data.

        Returns
        -------
        np.ndarray
            Transformed data after PCA.
        """

        pca = PCA(n_components=0.95, random_state=10)
        X_pca = pca.fit_transform(df)
        return X_pca

    @staticmethod
    def evaluate_model(X: np.ndarray, model_name: str, model, params: dict) -> Optional[dict]:
        """
        Evaluates a single model on the dataset using the given parameters.

        Parameters
        ----------
         X : np.ndarray
            Feature matrix.
        model_name : str
            Name of the clustering model.
        model : estimator
            Clustering model instance.
        params : dict
            Dictionary of parameters to set in the pipeline.

        Returns
        -------
        dict or None
            Dictionary with clustering labels and evaluation scores, or None if failed.
        с"""
        model = clone(model)
        pipeline = Pipeline([
                ('umap', UMAP()),
                (model_name, model)])
        pipeline.set_params(**params)
        try:
            labels = pipeline.fit_predict(X)
            score_silhouette = silhouette_score(X, labels)
            score_db = davies_bouldin_score(X, labels)
            return {
                'params': params,
                'labels': labels,
                'score_silhouette': score_silhouette,
                'score_db': score_db
                }
        except Exception as er:
            print(f'error: {er}')
            return None
        
    @staticmethod
    def evaluate_gridsearch(X: np.ndarray, models: dict, param_grid: dict) -> dict:
        """
        Performs a parallel grid search over all model configurations.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix.
        models : dict
            Dictionary of clustering model instances.
        param_grid : dict
            Dictionary mapping model names to their parameter grids.

        Returns
        -------
        dict
            Dictionary with results for each model.
        """
        results = {}
        for model_name, model in models.items():
            print(model_name)
            results[model_name] = Parallel(n_jobs=-1, verbose=10)(
            delayed(GridSearch.evaluate_model)(X, model_name, model, params)
            for params in ParameterGrid(param_grid[model_name])
            )
        return results
    
    @staticmethod
    def clean_results(results: dict) -> dict:
        """
        Removes None entries from the grid search results.

        Parameters
        ----------
        results : dict
            Dictionary with grid search results.

        Returns
        -------
        dict
            Filtered dictionary containing only successful evaluations.
        """
        clean_results = {}
        for model_name, model_results in results.items():
            if model_results:
                res = [r for r in model_results if r is not None]
                if res: 
                    clean_results[model_name] = res
        return clean_results
    
    @staticmethod
    def find_best_pipeline(results: dict, type: str) -> dict:
        """
        Finds the best pipeline for each model based on a scoring metric.

        Parameters
        ----------
        results : dict
            Cleaned result dictionary from grid search.
        type : {'max', 'min'}
            Whether to maximize silhouette score ('max') or minimize DB score ('min').

        Returns
        -------
        dict
            Dictionary with best pipeline info for each model.
        """
        agg_func = {
            'max': ('score_silhouette', max),
            'min': ('score_db', min)
        }

        clean_results = GridSearch.clean_results(results)

        try:
            best_pipeline = {
                model_name: agg_func[type][1](clean_results[model_name], key= lambda r: r[agg_func[type][0]]) \
                for model_name in clean_results 
                }
            return best_pipeline
        except Exception as ex:
            print(f'None in results: {ex}')
            return {}
    
    def search_params(self, pca_flag: bool) -> tuple[dict, dict]:
        """
        Runs the full grid search for all models and parameters.

        Parameters
        ----------
        pca_flag : bool
            Whether to apply PCA before UMAP.

        Returns
        -------
        tuple of dict
            (Best pipelines by silhouette, best by DB index).
        """
        if pca_flag:
            X = GridSearch.evaluate_pca(self.df)
        else: 
            X = self.df

        param_grid = {key: {**self.param_umap, **self.param_models[key]} for key in self.param_models}
        print(f'params: \n {param_grid}')

        results = GridSearch.evaluate_gridsearch(X, self.models, param_grid)

        best_silhouette = GridSearch.find_best_pipeline(results, 'max')
        best_db = GridSearch.find_best_pipeline(results, 'min')

        return best_silhouette, best_db
    

    def apply_pipeline_2D(self, df: pd.DataFrame, model_name: str, model, best_pipeline: dict) -> tuple[np.ndarray, np.ndarray]:
        """
        Applies UMAP projection to 2D and fits clustering model.

        Parameters
        ----------
        df : pandas.DataFrame
            Input feature matrix.
        model_name : str
            Name of the clustering model.
        model : estimator
            Clustering model instance.
        best_pipeline : dict
            Best parameters found in grid search.

        Returns
        -------
        tuple of np.ndarray
            (2D projection, clustering labels).
        """
        model = clone(model)
        params = best_pipeline['params']
        params.pop('umap__ncomponents', None)
        pipeline = Pipeline([
            ('umap', UMAP(n_components=2, random_state=42)),
            (model_name, model)
                ])
        pipeline.set_params(**params)
        pipeline.fit(df)
        processed = pipeline.named_steps['umap'].transform(df)
        labels = pipeline[-1].fit_predict(processed)
        return processed, labels

    @staticmethod
    def split_param(best_pipeline: dict, model_name: str) -> tuple[dict, dict]:
        """
        Separates UMAP and clustering parameters from a parameter dict.

        Parameters
        ----------
        best_pipeline : dict
            Dictionary containing pipeline info.
        model_name : str
            Clustering model name.

        Returns
        -------
        tuple of dict
            (UMAP parameters, clustering model parameters).
        """
        params = best_pipeline['params']
        params_umap = {}
        params_clustering = {}
        for key in params:
            if key.startswith('umap'):
                _, _, key_new = key.partition('__')
                params_umap[key_new] = params[key]
            else:
                _, _, key_new = key.partition('__')
                params_clustering[key_new] = params[key]
        params_umap.pop('n_components', None)
        return params_umap, params_clustering

    @staticmethod
    def visualize(ax, X: np.ndarray, labels: np.ndarray, model_cluster: str, pca_flag: bool) -> None:
        """
        Plots the 2D projection with clusters.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes on which to plot.
        X : np.ndarray
            2D projection of features.
        labels : np.ndarray
            Clustering labels.
        model_cluster : str
            Name of the clustering model.
        pca_flag : bool
            Whether PCA was applied before UMAP.
        """
            
        palette = plt.get_cmap('tab10')

        for label in set(labels):
            mask = labels == label
            color = 'gray' if label == -1 else palette(label % 10)
            ax.scatter(X[mask, 0], X[mask, 1], s=10, color=color, label=f'Cluster {label}' if label != -1 else 'Noise')
        if len(set(labels)) < 6:
            ax.legend()
        if pca_flag:
            ax.set_title(f'PCA + UMAP + {model_cluster.upper()}')
        else:
            ax.set_title(f'UMAP + {model_cluster.upper()}') #
        #ax.set_xlabel('UMAP-1 component')
        #ax.set_ylabel('UMAP-2  component')

    def visualize_best_score(self, ax, best_silhouette: dict, best_db: dict, pca_flag: bool) -> None:
        """
        Visualizes best clustering results for silhouette and DB metrics.

        Parameters
        ----------
        ax : np.ndarray of matplotlib.axes.Axes
            Grid of axes for plotting.
        best_silhouette : dict
            Best silhouette-based pipelines.
        best_db : dict
            Best DB-score-based pipelines.
        pca_flag : bool
            Whether PCA was applied before UMAP.
        """
         
        for j, pipe in enumerate((best_silhouette, best_db)):
            i = 0
            for model_name, model in self.models.items():
                processed, labels = self.apply_pipeline_2D(self.df, model_name, model, pipe[model_name])
                GridSearch.visualize(ax[j][i], processed, labels, model_name, pca_flag)
                i += 1

    def save_pipe_json(self, pipeline: dict, filename: str, path: str) -> None:
        """
        Saves the best pipeline results to a JSON file.

        Parameters
        ----------
        pipeline : dict
            Dictionary of best pipelines.
        filename : str
            Base filename without extension.
        path : str
            Directory where to save the file.
        """
        serializable = {}
        for model_name, result in pipeline.items():
            serializable[model_name] = {
                'params': result.get('params', {}),
                'score_silhouette': result.get('score_silhouette'),
                'score_db': result.get('score_db')
                }
        try:
            with open(path+filename+'.json', 'w') as f:
                json.dump(serializable, f, indent=4)
        except Exception as ex:
            print(f'Error during writing data to a file: {ex}')

    def save_pipe_pickle(self, pipeline: dict, filename: str, path: str) -> None:
        """
        Saves the pipeline dictionary to a pickle file.

        Parameters
        ----------
        pipeline : dict
            Dictionary of best pipelines.
        filename : str
            Base filename without extension.
        path : str
            Directory where to save the file.
        """
        with open(path+filename+'.pkl', 'wb') as f:
            pickle.dump(pipeline, f)

    def save_pipe(self, pipeline: dict, filename: str) -> None:
        """
        Saves the pipeline to both JSON and pickle formats under default path.

        Parameters
        ----------
        pipeline : dict
            Dictionary of best pipelines.
        filename : str
            Base filename without extension.
        """
        path = './results/gridsearch/'
        os.makedirs(path, exist_ok=True)
        self.save_pipe_json(pipeline, filename, path)
        self.save_pipe_pickle(pipeline, filename, path)

    def load_pipe_json(self, filename: str) -> Optional[dict]:
        """
        Loads pipeline configuration from a JSON file.

        Parameters
        ----------
        filename : str
            Name of the JSON file (without extension).

        Returns
        -------
        dict or None
            Loaded pipeline, or None on failure.
        """
        try:
            with open('./results/gridsearch/'+filename+'.json', 'r') as openfile:
                openfile = json.load(openfile)
            return openfile
        except Exception as ex:
            print(f'Error during reading data to a file: {ex}')
            return None
        
    def load_pipe_pickle(self, filename: str) -> dict:
        """
        Loads pipeline data from a pickle file.

        Parameters
        ----------
        filename : str
            Name of the pickle file (without extension).

        Returns
        -------
        dict
            Loaded pipeline dictionary.
        """
        with open('./results/gridsearch/'+filename+'.pkl', 'rb') as f:
            return pickle.load(f)


