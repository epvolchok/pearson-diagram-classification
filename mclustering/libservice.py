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
from typing import Optional, Tuple
import logging
logger = logging.getLogger(__name__)

class ServiceFuncs:

    """
    A static utility class providing helper functions for file handling, 
    image validation, and folder management.

    This class is not meant to be instantiated.

    """

    def __init__(self):
        ServiceFuncs.init_error(__class__.__name__)
    
    @staticmethod
    def init_error(name: str) -> None:
        """Raise an error to prevent instantiation of static utility classes."""
        logger.error(RuntimeError(f'This class [{name}] can not be instantiate.'))
        raise RuntimeError(f'This class [{name}] can not be instantiate.')

    @staticmethod
    def check_extension(file_path: str, allowed_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')) -> bool:
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
        
        return file_path.lower().endswith(allowed_extensions)
    
    
    @staticmethod
    def extract_observ_data(path: str, pattern: str = r'((?:r|t))swf-e_(\d{4})(\d{2})(\d{2})') -> Tuple[str, datetime.date]:
        """
        Extract observation type and date from the filename using regex.

        Parameters
        ----------
        path : str
            File path or name.
        pattern : str
            Regular expression pattern to extract observation type and date.

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
    def preparing_folder(dir_name: str, clear: bool) -> None:
        """
        Prepare a directory for output, optionally clearing its contents.

        Parameters
        ----------
        dir_name : str
            Path to the target directory.
        clear : bool
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
    def input_name(input_path: str, pattern: str = r'images_(\w+)') -> str:
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
        search = re.search(pattern, input_path)
        if search:
            return search.group(1)
        else:
            return input_path

    @staticmethod
    def create_name(default_filename: str, file_to_write: Optional[str] = None, suffix: Optional[str] = None) -> str:
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
    A static utility class providing helper functions for database management,
    including loading, saving, and processing metadata.

    This class is not meant to be instantiated.
    """
    def __init__(self):
        ServiceFuncs.init_error(__class__.__name__)

    @staticmethod
    def load_info(info_path: str) -> pd.DataFrame:
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
    def save_database(df: pd.DataFrame, file_to_write: Optional[str] = None, kind: str = 'pickle') -> None:
        """
        Save a pandas DataFrame to disk.

        Parameters
        ----------
        df : pandas.DataFrame
            Data to save.
        file_to_write : str, optional
            Base filename without extension.
        kind : {'pickle', 'json'}
            Format to save the file in.
        """
        default_name = os.path.join(os.getcwd(), 'results', 'pearson_diagram_data')
        file_to_write = ServiceFuncs.create_name(default_name, file_to_write=file_to_write)

        #print(f'Saving the database to {file_to_write}')
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
    def read_database(file_to_read: Optional[str] = None, kind: str = 'pickle',
                      dtype: Optional[dict] = {'obsertype': 'category', 'label': 'category', 'date': 'datetime'}) -> Optional[pd.DataFrame]:
        """
        Read a pandas DataFrame from a file.

        Parameters
        ----------
        file_to_read : str, optional
            Base filename without extension.
        kind : {'pickle', 'json'}
            Format of the file.
        dtype : dict, optional
            Column types to enforce when reading JSON.

        Returns
        -------
        pandas.DataFrame or None
            The loaded DataFrame, or None if reading failed.
        """
        default_name = os.path.join(os.getcwd(), 'results', 'pearson_diagram_data')
        file_to_read = ServiceFuncs.create_name(default_name, file_to_write=file_to_read)
        
        #print(f'Reading a database from a file {file_to_read}')
        logger.info(f'Reading a database from a file {file_to_read}')
        try:
            if kind == 'pickle':
                df = pd.read_pickle(file_to_read+'.pkl')
            elif kind == 'json':
                df = pd.read_json(file_to_read+'.json', dtype=dtype)
            return df
        except Exception as e:
            print(f'Error during reading a database: {e}')
            logger.error(f'Error during reading a database: {e}')
            return None
        
    @staticmethod
    def split_into_two(df: pd.DataFrame, excluded_columns: Optional[list] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
            (df_features, df_metadata)
        """
        if excluded_columns is None:
            excluded_columns=['dataset_name', 'date', 'dist_to_sun[au]', 'SAMPLES_NUMBER', 
                          'SAMPLING_RATE[kHz]', 'SAMPLE_LENGTH[ms]']
        for possible_path in ['path', 'oldpath']:
            if possible_path in df.columns:
                if 'path' not in excluded_columns and 'oldpath' not in excluded_columns:
                    excluded_columns = excluded_columns + [possible_path]
                break

        mask = [cols for cols in df.columns if cols not in excluded_columns]
        df_features = df.loc[:, mask]
        df_metadata = df.loc[:, excluded_columns]
        return df_features, df_metadata

class Logg:
    """
    A utility class for managing logging setup and log file naming.

    This class is not meant to be instantiated.
    """

    def __init__(self):
        ServiceFuncs.init_error(__class__.__name__)

    @staticmethod
    def get_log_filename(log_dir: str = 'logs', log_name: Optional[str] = None) -> str:
        """
        Determines a log file name.

        Parameters
        ----------
        log_dir : str
            Directory where log files are stored.
        log_name : str or None
            Custom log file name. If None, a new sequential name will be generated.

        Returns
        -------
        str
            Full path to the log file.
        """
        os.makedirs(log_dir, exist_ok=True)

        if log_name:
            if not log_name.endswith('.log'):
                log_name += '.log'
            return os.path.join(log_dir, log_name)

        # Get existing log files
        existing_logs = [f for f in os.listdir(log_dir) if f.startswith('log_') and f.endswith('.log')]
        count = len(existing_logs)
        next_num = count + 1

        if count < 10:
            name = f'log_{next_num:02d}.log'  # log_01.log, log_02.log, ...
        else:
            name = f'log_{next_num}.log'      # log_11.log, log_12.log, ...

        return os.path.join(log_dir, name)

    @staticmethod
    def setup_logging(log_file: str = 'pipeline.log') -> None:
        """
        Set up logging configuration.

        Parameters
        ----------
        log_file : str
            Path to the log file.
        """
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                #logging.StreamHandler()
            ]
        )