#Copyright (c) 2025 Evgeniia VOLCHOK
#for contacts e.p.volchok@gmail.com

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0

import os
from mclustering import*
import pandas as pd
from typing import Optional, Tuple
import logging
logger = logging.getLogger(__name__)

class InputManager:
    """
    A static utility class for handling interactive user input in the console.

    Provides methods for:
    - Displaying welcome messages and prompts
    - Asking yes/no questions with validation
    - Parsing dictionaries from key=value input
    - Converting string input to appropriate Python types (bool, int, float, str)
    - Confirming or modifying regular expressions for dataset matching

    This class is not meant to be instantiated.
    """

    def __init__(self):
        ServiceFuncs.init_error(__class__.__name__)
    
    @staticmethod
    def input_wrapper(message: str) -> Optional[str]:
        """
        Prompts the user for input and handles KeyboardInterrupt gracefully.

        Parameters
        ----------
        message : str
            Prompt message to display to the user.

        Returns
        -------
        str or None
            User input string or None if interrupted.
        """
        try:
            user_input = input(message)
            return user_input
        except KeyboardInterrupt:
            logger.error('KeyboardInterrupt')
            print('\nInterrupted by user.')
            return
        
    @staticmethod
    def welcome_message() -> None:
        """
        Prints a welcome message with initial instructions for placing image data.
        """

        message = 'Welcome to the script! \n' + \
        'First, put your images in a separate folder (in "./images/") named according to the pattern: \n \
        "images_(your_specification)"'
        print(message)
        logger.info('Script started')

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
                    logger.error(f'Invalid input, please enter True or False! {e}')
            else:
                return True
                

    @staticmethod
    def filter_mixed_freq(name_pattern: str) -> Tuple[bool, str]:
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
        logger.info(f'Filter mixed frequencies: {filter_mixed}')
        if filter_mixed:
            print(f'Check the name pattern: {name_pattern}')
            np_input = InputManager.input_wrapper('Change if it is needed or press "Enter" > ').strip()
            if np_input:
                name_pattern = np_input
        logger.info(f'Name pattern: {name_pattern}')
        return filter_mixed, name_pattern
    
    @staticmethod
    def input_dict(prompt: str = 'Enter pairs of key=value (empty string - end) > ') -> dict:
        """
        Interactively builds a dictionary from user input of key=value pairs.

        Parameters
        ----------
        prompt : str
            Message shown before input begins.

        Returns
        -------
        dict
            Dictionary parsed from input.
        """
        print(prompt)
        result = {}
        while True:
            line = InputManager.input_wrapper(' > ').strip()
            if not line:
                break
            if '=' not in line:
                print('Wrong format, Enter pairs of key=value')
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = InputManager.parse_value(value.strip())
            result[key] = value
        return result
    
    @staticmethod
    def parse_value(value: str) -> object:
        """
        Attempts to convert a string to a boolean, integer, float, or returns as string.

        Parameters
        ----------
        value : str
            Input string to parse.

        Returns
        -------
        object
            Parsed value (bool, int, float, or str).
        """
        
        value = value.strip()
        if value.lower() in {'true', 'yes'}:
            return True
        elif value.lower() in {'false', 'no'}:
            return False
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value
    
    @staticmethod
    def check_if_database(obj):
        if isinstance(obj, ResNetFeatures):
            df = obj.database
        elif isinstance(obj, Clustering):
            df = obj.df
        elif isinstance(obj, pd.DataFrame):
            df = obj
        else:
            logger.error(ValueError('(features) is non-known object'))
            raise ValueError('(features) is non-known object')
        return df



class PathManager:
    """
    A static utility class for managing file paths, folders, and output saving.

    Provides methods for:
    - Prompting and validating image folder names
    - Creating result and base directories
    - Asking for custom or default paths
    - Saving pandas DataFrames with user-defined filenames

    This class is not meant to be instantiated.
    """
    def __init__(self):
        ServiceFuncs.init_error(__class__.__name__)
    
    @staticmethod
    def preparations() -> Tuple[str, str, str]:
        """
        Sets up required working directories and obtains the user's image folder name.

        Returns
        -------
        input_imags : str
            Name of the subdirectory in 'images/' containing image data.
        default_filename : str
            Base filename used for saving feature-related data.
        results_dir : str
            Directory for processed results.
        """

        base_dir_names = ['images', 'processed', 'figures', 'data', 'results', 'logs']
        cwd = os.getcwd()
        message = 'Use "processed" folder for processed images \n' + \
        '"data" folder for info metadata \n' + \
        '"figures" and "results" for graphic and data results, respectively.'
        print(message)
        
        PathManager.create_base_dirs(*base_dir_names)

        input_imags = PathManager.images_folder()
        default_filename, results_dir = PathManager.create_results_dir(input_imags)
        
        return input_imags, default_filename, results_dir
    
    @staticmethod
    def create_base_dirs(*args: str) -> None:
        """
        Ensures that a set of base directories exists (creates if missing).

        Parameters
        ----------
        *args : str
            Names of directories to ensure exist within the current working directory.
        """
        cwd = os.getcwd()
        for dirname in args:
            dirname = os.path.join(cwd, dirname)
            ServiceFuncs.preparing_folder(dirname, clear=False)
        logger.info('Base directories checked or created')
    
    @staticmethod
    def images_folder() -> str:
        """
        Prompts the user to input a valid subdirectory name inside 'images/'.

        Repeats until a valid directory is given.

        Returns
        -------
        str
            Valid image folder name.
        """
        cwd = os.getcwd()
        input_imags = os.path.basename(InputManager.input_wrapper('Please enter the name of directory with images > '))
        isdir = os.path.isdir(os.path.join(cwd, 'images', input_imags))

        while not isdir:
            print(f'Wrong folder name: {input_imags}. Try again.')
            logger.debug(f'Wrong folder name: {input_imags}')
            input_imags = os.path.basename(InputManager.input_wrapper('Please enter the name of directory with images > '))
            isdir = os.path.isdir(os.path.join(cwd, 'images', input_imags))

        return input_imags
    
    @staticmethod
    def create_results_dir(input_imags: str) -> Tuple[str, str]:
        """
        Constructs paths for results and prompts whether to clear previous outputs.

        Parameters
        ----------
        input_imags : str
            Image folder name, used to derive output paths.

        Returns
        -------
        default_filename : str
            Base path for saving output data.
        results_dir : str
            Directory for saving processed files.
        """
        cwd = os.getcwd()
        specification = ServiceFuncs.input_name(input_imags)
        results_dir = os.path.join(cwd, 'processed', 'processed_'+specification)
        default_filename = os.path.join(cwd, 'results', 'pearson_diagram_data_'+specification)
        logger.info(f'Source images: {input_imags} Results: {results_dir, default_filename}')

        clear = InputManager.get_bool(f'If folder {results_dir} already exists would you like to clear its contents? (True or False) > ')
        ServiceFuncs.preparing_folder(results_dir, clear=clear)
        logger.info(f'Clearing results folder: {clear}')
        return default_filename, results_dir

    @staticmethod
    def get_path(message: str, default_path: str) -> str:
        """
        Prompts the user for a path or returns the default if input is empty.

        Parameters
        ----------
        message : str
            Prompt to display.
        default_path : str
            Path to use if user provides nothing.

        Returns
        -------
        str
            The path chosen by the user.
        """

        file_path = InputManager.input_wrapper(message)
        file_path = file_path if file_path else default_path
        return file_path
    

    
    @staticmethod
    def saving_database(df: pd.DataFrame, default_filename: str, message: str, suf: str) -> None:
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
        logger.info(f'Saving database')
        file_to_write = InputManager.input_wrapper(message)
        suffix = InputManager.input_wrapper('or/and enter a suffix to the filename > ') or suf
        file_to_write = ServiceFuncs.create_name(default_filename, file_to_write=file_to_write, suffix=suffix)
        DBFuncs.save_database(df, file_to_write=file_to_write)




