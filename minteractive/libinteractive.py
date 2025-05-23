#Copyright (c) 2025 Evgeniia VOLCHOK
#for contacts e.p.volchok@gmail.com

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0

import os
from mclustering import*

import logging
logger = logging.getLogger(__name__)

class InputManager:
    def __init__(self):
        ServiceFuncs.init_error(__class__.__name__)
    
    @staticmethod
    def input_wrapper(message):
        try:
            user_input = input(message)
            return user_input
        except KeyboardInterrupt:
            logger.error('KeyboardInterrupt')
            print('\nInterrupted by user.')
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
        logger.info(f'Filter mixed frequencies: {filter_mixed}')
        if filter_mixed:
            print(f'Check the name pattern: {name_pattern}')
            np_input = InputManager.input_wrapper('Change if it is needed or press "Enter": ').strip()
            if np_input:
                name_pattern = np_input
        logger.info(f'Name pattern: {name_pattern}')
        return filter_mixed, name_pattern
    
    def input_dict(prompt='Enter pairs of key=value (empty string - end):'):
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
    
    def parse_value(value: str):
        
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



class PathManager:
    def __init__(self):
        ServiceFuncs.init_error(__class__.__name__)
    
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
        cwd = os.getcwd()
        message = 'Use "processed" folder for processed images \n' + \
        '"data" folder for info metadata \n' + \
        '"figures" and "results" for graphic and data results, respectively.'
        print(message)
        
        PathManager.create_base_dirs(*base_dir_names)

        input_imags = PathManager.images_folder()
        default_filename = PathManager.create_results_dir(input_imags)
        
        return input_imags, default_filename
    
    @staticmethod
    def create_base_dirs(*args):
        cwd = os.getcwd()
        for dirname in args:
            dirname = os.path.join(cwd, dirname)
            ServiceFuncs.preparing_folder(dirname, clear=False)
        logger.info('Base directories checked or created')
    
    @staticmethod
    def images_folder():
        cwd = os.getcwd()
        input_imags = os.path.basename(InputManager.input_wrapper('Please enter the name of directory with images: '))
        isdir = os.path.isdir(os.path.join(cwd, 'images', input_imags))

        while not isdir:
            print(f'Wrong folder name: {input_imags}. Try again.')
            logger.debug(f'Wrong folder name: {input_imags}')
            input_imags = os.path.basename(InputManager.input_wrapper('Please enter the name of directory with images: '))
            isdir = os.path.isdir(os.path.join(cwd, 'images', input_imags))

        return input_imags
    
    @staticmethod
    def create_results_dir(input_imags):
        cwd = os.getcwd()
        specification = ServiceFuncs.input_name(input_imags)
        results_dir = os.path.join(cwd, 'processed', 'processed_'+specification)
        default_filename = os.path.join(cwd, 'results', 'pearson_diagram_data_'+specification)
        logger.info(f'Source images: {input_imags} Results: {results_dir, default_filename}')

        clear = InputManager.get_bool(f'If folder {results_dir} already exists would you like to clear its contents? (True or False) ')
        ServiceFuncs.preparing_folder(results_dir, clear=clear)
        logger.info(f'Clearing results folder: {clear}')
        return default_filename

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
        logger.info(f'Saving database')
        file_to_write = InputManager.input_wrapper(message)
        suffix = InputManager.input_wrapper('or/and enter a suffix to the filename: ') or suf
        file_to_write = ServiceFuncs.create_name(default_filename, file_to_write=file_to_write, suffix=suffix)
        DBFuncs.save_database(df, file_to_write=file_to_write)




