#Copyright (c) 2025 Evgeniia VOLCHOK
#for contacts e.p.volchok@gmail.com

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0


import pandas as pd
import os
import re
import datetime
import shutil

import logging
logger = logging.getLogger(__name__)

class ServiceFuncs:

    """
    A static utility class providing helper functions for file handling, 
    image validation, and folder management.

    This class is not meant to be instantiated.

    Methods
    -------
    check_extension(file_path, allowed_extensions)
        Validates the file extension against allowed types.

    load_info(info_path)
        Loads and processes observation metadata from a text file.

    extract_observ_data(path, pattern)
        Extracts observation type and date from a file path using regex.

    save_database(df, file_to_write, kind)
        Saves a pandas DataFrame to a file in pickle or JSON format.

    read_database(file_to_read, kind)
        Reads a pandas DataFrame from a file.

    preparing_folder(dir_name, clear)
        Prepares a directory, optionally clearing its contents.

    split_into_two(df, excluded_columns)
        Splits a DataFrame into features and metadata parts.

    input_name(input_path, pattern)
        Extracts a name ID from a path using regex.
    """

    def __init__(self):
        ServiceFuncs.init_error()
    
    @staticmethod
    def init_error():
        logger.error(RuntimeError(f'This class [{__class__.__name__}] can not be instantiate.'))
        raise RuntimeError(f'This class [{__class__.__name__}] can not be instantiate.')

    @staticmethod
    def check_extension(file_path, allowed_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        """
        Check if a file has one of the allowed extensions.

        Parameters
        ----------
        file_path : str
            Path to the file.
        allowed_extensions : tuple of str, optional
            Tuple of allowed file extensions.

        Returns
        -------
        bool
            True if the file has an allowed extension, False otherwise.
        """
        if not file_path.lower().endswith(allowed_extensions):
            return False
        return True
    
    
    @staticmethod
    def extract_observ_data(path, pattern=r'(?:r|t)swf-e_(\d{4})(\d{2})(\d{2})'):
        """
        Extracts observation type and date from the filename using regex.

        Parameters
        ----------
        path : str
            File path or name.

        Returns
        -------
        tuple
            (observation_type, datetime.date)
        
        Raises
        ------
        ValueError
            If the pattern is not found.
        """
        search = re.search(pattern, path)
        if not search:
            logger.error(ValueError(f'Could not extract observation data from: {path}'))
            raise ValueError(f'Could not extract observation data from: {path}')
        obs_type = search.group(1)
        date = datetime.date(int(search.group(2)), int(search.group(3)), int(search.group(4)))
        return obs_type, date

        
    @staticmethod
    def preparing_folder(dir_name, clear):
        """
        Prepare a directory for output, optionally clearing its contents.

        Parameters
        ----------
        dir_name : str
            Path to the target directory.
        clear : bool, optional
            If True, the directory will be deleted and recreated.

        Raises
        ------
        Exception
            If directory manipulation fails.
        """
        if os.path.exists(dir_name):
            if clear:
                try:
                    shutil.rmtree(dir_name)
                    os.makedirs(dir_name)
                except Exception as e:
                    print(f'Error during deleting files in {dir_name}: {e}')
                    logger.error(f'Error during deleting files in {dir_name}: {e}')
        else:
            os.makedirs(dir_name)


    @staticmethod
    def input_name(input_imags, pattern=r'images_(\w+)'):
        """
        Extract an identifier from a file path using a regular expression.

        Parameters
        ----------
        input_path : str
            File path or name.
        pattern : str
            Regular expression to extract the identifier.

        Returns
        -------
        str
            Extracted identifier or the original input if no match found.
        """
        search = re.search(pattern, input_imags)
        if search:
            return search.group(1)
        else:
            return input_imags

    @staticmethod
    def create_name(default_filename, file_to_write=None, suffix=None):
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
    
class DBFuncs:
    """
    A static utility class providing helper for a database management.
    This class is not meant to be instantiated.
    """
    def __init__(self):
        ServiceFuncs.init_error()

    @staticmethod
    def load_info(info_path):
        """
        Load and preprocess metadata from a space-delimited text file.

        Parameters
        ----------
        info_path : str
            Path to the metadata file.

        Returns
        -------
        pandas.DataFrame
            Parsed and enriched metadata, including date and sampling rate in kHz.
        """
        logger.info(f'Loading observation parameters description from {info_path}')
        info = pd.read_csv(info_path, delimiter=' ')
        info['date'] = pd.to_datetime(info[['year', 'month', 'day']])
        info['SAMPLING_RATE[kHz]'] = info['SAMPLING_RATE[Hz]'].floordiv(1000)
        info.drop('SAMPLING_RATE[Hz]', axis=1, inplace=True)
        return info
    
    @staticmethod
    def save_database(df, file_to_write=None, kind='pickle'):
        """
        Save a pandas DataFrame to disk.

        Parameters
        ----------
        df : pandas.DataFrame
            Data to save.
        file_to_write : str
            Base filename without extension.
        kind : {'pickle', 'json'}
            Format to save the file in.
        save_dir : str
            Directory to save the file to.

        Raises
        ------
        Exception
            If saving fails.
        """
        default_name = os.path.join(os.getcwd(), 'results', 'pearson_diagram_data')
        file_to_write = ServiceFuncs.create_name(default_name, file_to_write=file_to_write)

        print(f'Saving the database to {file_to_write}')
        logger.info(f'Saving the database to {file_to_write}')
        try:
            if kind == 'pickle':
                df.to_pickle(file_to_write+'.pkl')
            elif kind == 'json':
                df.to_json(file_to_write+'.json')
        except Exception as e:
            print(f'Failed to save the database: {e}')
            logger.error(f'Failed to save the database: {e}')

    @staticmethod
    def read_database(file_to_read=None, kind='pickle', 
                dtype={'obsertype': 'category', 'label': 'category', 'date': 'datetime'}):
        """
        Read a pandas DataFrame from a file.

        Parameters
        ----------
        file_to_read : str
            Base filename without extension.
        kind : {'pickle', 'json'}
            Format of the file.
        dtype : dict
            To set types of columns in the database.

        Returns
        -------
        pandas.DataFrame or None
            The loaded DataFrame, or None if reading failed.
        """
        default_name = os.path.join(os.getcwd(), 'results', 'pearson_diagram_data')
        file_to_read = ServiceFuncs.create_name(default_name, file_to_write=file_to_read)
        
        print(f'Reading a database from a file {file_to_read}')
        logger.info(f'Reading a database from a file {file_to_read}')
        try:
            if kind == 'pickle':
                df = pd.read_pickle(file_to_read+'.pkl')
            elif kind == 'json':
                df = pd.read_json(file_to_read+'.json', dtype=dtype)
            print(df.head())
            print(df.dtypes)
            return df
        except Exception as e:
            print(f'Error during reading a database: {e}')
            logger.error(f'Error during reading a database: {e}')
            return None
        
    @staticmethod
    def split_into_two(df, excluded_columns=None):
        """
        Split a DataFrame into features and metadata columns.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame.
        excluded_columns : list of str, optional
            Columns to exclude from the features set.

        Returns
        -------
        tuple of pandas.DataFrame
            (features_df, metadata_df)
        """
        if excluded_columns is None:
            excluded_columns=['dataset_name', 'date', 'dist_to_sun[au]', 'SAMPLES_NUMBER', 
                          'SAMPLING_RATE[kHz]', 'SAMPLE_LENGTH[ms]', 'oldpath']
        mask = [cols for cols in df.columns if cols not in excluded_columns]
        df_features = df.loc[:, mask]
        excluded_part = df.loc[:, excluded_columns]
        return df_features, excluded_part

