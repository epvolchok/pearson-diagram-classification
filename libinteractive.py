import matplotlib.pyplot as plt
import os
import re

from libpreprocessing import FeaturesPreprocessing
from libclustering import Clustering
from libfeatures import ResNetFeatures
from libservice import ServiceFuncs

class InteractiveMode:
    def __init__(self):
        raise RuntimeError('This class can not be instantiate.')
    
    @staticmethod
    def welcome_message():
        message = 'Welcome to the script! \n' + \
        'First place your images in a separate folder (into "./images/") named according to the template: \n \
        "images_(your_specification)"'
        print(message)

    @staticmethod
    def preparations():
        base_dir_names = ['./images', './processed', './figures', './data', './results']
        message = 'Use "./processed" folder for processed images \n' + \
        '"./data" folder for info metadata \n' + \
        '"./figures" and "./results" for graphic and data results, respectively.'
        print(message)

        for dirname in base_dir_names:
            ServiceFuncs.preparing_folder(dirname, clear=False)

        input_imags = input('Please enter the name of your working directory: ').strip('./, ')
        specification = ServiceFuncs.input_name(input_imags)
        results_dir = './processed/processed_'+specification
        default_filename = './results/pearson_diagram_data_'+specification

        clear = InteractiveMode.get_bool(f'If folder {results_dir} already exists would you like to clear its contents? (True or False) ')
        ServiceFuncs.preparing_folder(results_dir, clear=clear)

        return input_imags, default_filename

    @staticmethod
    def get_bool(prompt: str)-> bool: 
        """
        Prompt the user for a boolean (True/False) input.

        Parameters
        ----------
        prompt : str
            The message to display.

        Returns
        -------
        bool
            The user's input as a boolean.

        Raises
        ------
        KeyError
            If the input is not 'true' or 'false' (case-insensitive).
        """
        while True:
            try:
                return {'true': True, 'false': False}[input(prompt).lower()]
            except KeyError as e:
                print(f'Invalid input, please enter True or False! {e}')

    @staticmethod
    def filter_mixed_freq(name_pattern):
        filter_mixed = InteractiveMode.get_bool('Would you like to filter mixed frequencies? (True or False) ')
        if filter_mixed:
            print(f'Check the name pattern: {name_pattern}')
            np_input = input('Change if it is needed or press "Enter": ').strip()
            if np_input:
                name_pattern = np_input
        return filter_mixed, name_pattern
    
    @staticmethod
    def get_path(message, default_path):
        file_path = input(message)
        file_path = file_path if file_path else default_path
        return file_path
    
    @staticmethod
    def get_features(input_imags, info_path, default_filename, name_pattern):
        message = f'To extract features from images located in {input_imags} enter 1. \n' + \
                'To load features from a file enter 2.'
        print(message)

        while True:
            choice = int(input('[1/2]: ').strip())

            if choice == 1:
                features = InteractiveMode.extract_features(input_imags, info_path, default_filename, name_pattern)
                break

            elif choice == 2:
                
                features = InteractiveMode.read_features(input_imags, info_path, default_filename, name_pattern)
                break
            else:
                print('Invalid input. \n Try again!')
        return features
    
    @staticmethod
    def extract_features(input_imags, info_path, default_filename, name_pattern):
        message = f'Enter a file name to write extracted features (or press "Enter" for default "{default_filename}"): '
        file_to_write = InteractiveMode.get_path(message, default_filename)
        filter_mixed, name_pattern = InteractiveMode.filter_mixed_freq(name_pattern)

        print('Features extraction')
        features = ResNetFeatures(
            path='./images/' + input_imags + '/',
            flag='extract',
            info_path=info_path,
            filter_mixed=filter_mixed,
            name_pattern=name_pattern,
            extra_params={'file_to_write': file_to_write}
        )
        return features
    
    @staticmethod
    def read_features(input_imags, info_path, default_filename, name_pattern):
        message = f'Enter a file name to read from (or press "Enter" for default "{default_filename}"): '
        file_to_read = InteractiveMode.get_path(message, default_filename)
        filter_mixed, name_pattern = InteractiveMode.filter_mixed_freq(name_pattern)

        print('Loading features')
        features = ResNetFeatures(
            path='./images/' + input_imags + '/',
            flag='read',
            info_path=info_path,
            filter_mixed=filter_mixed,
            name_pattern=name_pattern,
            extra_params={'file_to_read': file_to_read}
        )
        return features
    
    @staticmethod
    def run_processing(features, default_filename):
        message = 'Would you like to launch the standart processing algorithm (1) or \n' + \
                    'to call separate blocks of processing manually (2)?'
        print(message)
        while True:
            num=input('Enter 1 or 2 (or break to stop): ')
            if num == 'break':
                break
            else:
                if int(num) == 1:
                    InteractiveMode.standart_algorithm(features, default_filename)
                elif int(num) == 2:
                    print('This part has not been implemented yet. Use 1 or break.')



    @staticmethod
    def standart_algorithm(features, default_filename):
        message = 'The standart algorithm: \n' + \
            '1. Filtration of ResNet features by variance, thershold=1e-5 \n' + \
            '2. Preprocessing of the filtered features with \n' + \
            '   - PCA(n_components=0.95, svd_solver=\'full\') \n' + \
            '   - UMAP(n_components=20, min_dist=0.1, metric=\'cosine\') \n' + \
            '3. Clusterization \n' + \
            '- HDBSCAN(min_cluster_size=15, min_samples=5, metric=\'euclidean\') \n' + \
            '4. Visualization with PCA+UMAP2D+HDBSCAN'
        print(message)
        
        features = InteractiveMode.filtration(features)
        print(features)
        print(features.database)

        message = f'Enter a file name to save the filtered data (or press "Enter" to use default name): '
        InteractiveMode.saving_database(features.database, default_filename, message)

        processed = InteractiveMode.run_preprocessing(features, 'PCA+UMAPND')

        clusters = InteractiveMode.run_clusterization(processed)

        message = 'Enter a file name to save the clustered data (or press "Enter" to use default name): '
        InteractiveMode.saving_database(clusters.df, default_filename, message)

        # Visualization
        clusters.visualize_HDBSCAN(features.database)

    @staticmethod
    def filtration(features):
        print('Filtration')
        features.database = features.filtering_by_variance()
        features.info_on_features()
        return features
    
    @staticmethod
    def create_name_to_save(default_filename, file_to_write=None, suffix=None):
        if file_to_write is None and not suffix is None:
            file_to_write = default_filename + suffix
        elif file_to_write and suffix:
            file_to_write = file_to_write + suffix
        else:
            file_to_write = default_filename
        return file_to_write
    
    @staticmethod
    def saving_database(df, default_filename, message):
        file_to_write = input(message)
        suffix = input('or/and enter a suffix to the filename: ')
        file_to_write = InteractiveMode.create_name_to_save(default_filename, file_to_write=file_to_write, suffix=suffix)
        ServiceFuncs.save_database(df, file_to_write=file_to_write)

    @staticmethod
    def run_preprocessing(features, pipe_str='PCA+UMAPND'):
        print('Preprocessing')
        preprop = FeaturesPreprocessing(features)
        processed = preprop.wrapper_preprop(features.database, pipe_str)
        return processed
    
    @staticmethod
    def run_clusterization(features):
        print('Clusterization')
        clusters = Clustering(features)
        df_features, _ = ServiceFuncs.split_into_two(features)
        _, num_clusters = clusters.clustering_HDBSCAN(df_features)
        print(f'Number of clusters 20D: {num_clusters}')
        clusters.update_database()
        clusters.sort_files()
        return clusters
        

