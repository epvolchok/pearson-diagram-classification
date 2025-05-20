#Copyright (c) 2025 Evgeniia VOLCHOK
#for contacts e.p.volchok@gmail.com

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0
import os
from libpreprocessing import FeaturesPreprocessing
from libclustering import Clustering
from libfeatures import ResNetFeatures
from libservice import ServiceFuncs

class InputManager:
    def __init__(self):
        raise RuntimeError('This class can not be instantiate.')
    
    @staticmethod
    def input_wrapper(message):
        try:
            user_input = input(message)
            return user_input
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            return
        
    @staticmethod
    def welcome_message():
        """
        Prints a welcome message with initial instructions for placing image data.
        """

        message = 'Welcome to the script! \n' + \
        'First place your images in a separate folder (into "./images/") named according to the template: \n \
        "images_(your_specification)"'
        print(message)

    @staticmethod
    def get_bool(prompt: str)-> bool: 
        """
        Prompts the user for a True/False response.

        Parameters
        ----------
        prompt : str
            Prompt text for user input.

        Returns
        -------
        bool
            Boolean interpretation of the user's input.

        Raises
        ------
        KeyError
            If input is not 'true' or 'false' (case-insensitive).
        """
        while True:
            
            answ = InputManager.input_wrapper(prompt).lower()
            if answ:
                try:
                    return {'true': True, 'false': False}[answ]
                except KeyError as e:
                    print(f'Invalid input, please enter True or False! {e}')
            else:
                return True
                

    @staticmethod
    def filter_mixed_freq(name_pattern):
        """
        Asks the user whether to filter out mixed-frequency data.

        Parameters
        ----------
        name_pattern : str
            Default regular expression used to match dataset names.

        Returns
        -------
        filter_mixed : bool
            Whether to apply frequency filtering.
        name_pattern : str
            Possibly updated regex pattern provided by the user.
        """

        filter_mixed = InputManager.get_bool('Would you like to filter mixed frequencies? (True or False) ')
        if filter_mixed:
            print(f'Check the name pattern: {name_pattern}')
            np_input = InputManager.input_wrapper('Change if it is needed or press "Enter": ').strip()
            if np_input:
                name_pattern = np_input
        return filter_mixed, name_pattern

class PathManager:
    def __init__(self):
        raise RuntimeError('This class can not be instantiate.')
    
    @staticmethod
    def preparations():
        """
        Prepares required working directories and gathers the user’s image folder name.

        Returns
        -------
        input_imags : str
            Name of the user-specified image subdirectory.
        default_filename : str
            Suggested default filename for saving feature data.
        """

        base_dir_names = ['images', 'processed', 'figures', 'data', 'results', 'logs']
        message = 'Use "processed" folder for processed images \n' + \
        '"data" folder for info metadata \n' + \
        '"figures" and "results" for graphic and data results, respectively.'
        print(message)
        cwd = os.getcwd()
        for dirname in base_dir_names:
            dirname = os.path.join(cwd, dirname)
            ServiceFuncs.preparing_folder(dirname, clear=False)

        input_imags = os.path.basename(InputManager.input_wrapper('Please enter the name of your working directory: '))
        specification = ServiceFuncs.input_name(input_imags)
        results_dir = os.path.join(cwd, 'processed', 'processed_'+specification)
        default_filename = os.path.join(cwd, 'results', 'pearson_diagram_data_'+specification)

        clear = InputManager.get_bool(f'If folder {results_dir} already exists would you like to clear its contents? (True or False) ')
        ServiceFuncs.preparing_folder(results_dir, clear=clear)

        return input_imags, default_filename
    
    @staticmethod
    def get_path(message, default_path):
        """
        Gets a file path from the user, or uses the default if none is entered.

        Parameters
        ----------
        message : str
            Prompt for the user.
        default_path : str
            Default path to return if user provides no input.

        Returns
        -------
        str
            Final path.
        """

        file_path = InputManager.input_wrapper(message)
        file_path = file_path if file_path else default_path
        return file_path
    
    @staticmethod
    def create_name_to_save(default_filename, file_to_write=None, suffix=None):
        """
        Constructs a filename by combining a base name with a suffix.

        Parameters
        ----------
        default_filename : str
            Base filename.
        file_to_write : str, optional
            Custom filename provided by the user.
        suffix : str, optional
            Suffix to append to the filename.

        Returns
        -------
        str
            Full filename with optional suffix.
        """
        if not file_to_write:
            file_to_write = default_filename
        if suffix:
            file_to_write += suffix
        return file_to_write
    
    @staticmethod
    def saving_database(df, default_filename, message, suf):
        """
        Prompts the user to save the DataFrame to a file with optional suffix.

        Parameters
        ----------
        df : pandas.DataFrame
            Data to be saved.
        default_filename : str
            Default file base name.
        message : str
            Prompt shown to the user.
        suf : str
            Default suffix to append to the filename.
        """
        file_to_write = InputManager.input_wrapper(message)
        suffix = InputManager.input_wrapper('or/and enter a suffix to the filename: ') or suf
        file_to_write = PathManager.create_name_to_save(default_filename, file_to_write=file_to_write, suffix=suffix)
        ServiceFuncs.save_database(df, file_to_write=file_to_write)

class ProcessingPipeline:
    """
    This class serves as a high-level command-line wrapper around the functionality defined
    in modules like `libfeatures`, `libpreprocessing`, and `libclustering`. It guides the
    user through a series of prompts to configure and execute data processing operations.

    This class is not intended to be instantiated.
    """
    def __init__(self):
        raise RuntimeError('This class can not be instantiate.')
    
    @staticmethod
    def get_features(input_imags, info_path, default_filename, name_pattern):
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

        while True:
            try:
                choice = int(InputManager.input_wrapper('[1/2]: ').strip())
            except ValueError:
                print("Invalid input. Please enter 1 or 2.")
                continue

            if choice == 1:
                features = ProcessingPipeline.extract_features(input_imags, info_path, default_filename, name_pattern)
                break

            elif choice == 2:
                
                features = ProcessingPipeline.read_features(input_imags, info_path, default_filename, name_pattern)
                break
            else:
                print('Invalid input. \n Try again!')
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
        message = f'Enter a file name to write extracted features (or press "Enter" for default "{default_filename}"): '
        file_to_write = PathManager.get_path(message, default_filename)
        filter_mixed, name_pattern = InputManager.filter_mixed_freq(name_pattern)
        cwd = os.getcwd()
        print('Features extraction')
        features = ResNetFeatures(
            path=os.path.join(cwd, 'images', input_imags),
            flag='extract',
            info_path=info_path,
            filter_mixed=filter_mixed,
            name_pattern=name_pattern,
            extra_params={'file_to_write': file_to_write}
        )
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

        message = f'Enter a file name to read from (or press "Enter" for default "{default_filename}"): '
        file_to_read = PathManager.get_path(message, default_filename)
        filter_mixed, name_pattern = InputManager.filter_mixed_freq(name_pattern)
        cwd = os.getcwd()
        print('Loading features')
        features = ResNetFeatures(
            path=os.path.join(cwd, 'images', input_imags),
            flag='read',
            info_path=info_path,
            filter_mixed=filter_mixed,
            name_pattern=name_pattern,
            extra_params={'file_to_read': file_to_read}
        )
        return features
    
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
        while True:
            num=InputManager.input_wrapper('Enter 1 or 2 (or break to stop): ')
            if num == 'break':
                break
            else:
                if int(num) == 1:
                    ProcessingPipeline.standard_algorithm(features, default_filename)
                elif int(num) == 2:
                    print('This part has not been implemented yet. Use 1 or break.')



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
        
        features = ProcessingPipeline.filtration(features)

        message = f'Enter a file name to save the filtered data (or press "Enter" to use default name): '
        PathManager.saving_database(features.database, default_filename, message, suf='_filtered')

        processed = ProcessingPipeline.run_preprocessing(features, 'PCA+UMAPND')

        clusters = ProcessingPipeline.run_clusterization(processed)

        message = 'Enter a file name to save the clustered data (or press "Enter" to use default name): '
        PathManager.saving_database(clusters.df, default_filename, message, suf='_clustered')

        # Visualization
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
        features.database = features.filtering_by_variance()
        features.info_on_features()
        return features
    
    @staticmethod
    def run_preprocessing(features, pipe_str='PCA+UMAPND'):
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
        preprop = FeaturesPreprocessing(features)
        processed = preprop.wrapper_preprop(features.database, pipe_str)
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
        clusters = Clustering(features)
        df_features, _ = ServiceFuncs.split_into_two(features)
        _, num_clusters = clusters.clustering_HDBSCAN(df_features)
        print(f'Number of clusters 20D: {num_clusters}')
        clusters.update_database()
        clusters.sort_files()
        return clusters


