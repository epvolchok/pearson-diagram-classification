import pandas as pd
import numpy as np
import os
from PIL import Image
import re
import datetime

class ServiceFuncs:

    def __init__(self):
        raise RuntimeError("This class can not be instantiate.")
    
    @staticmethod
    def check_extension(file_path):
        allowed_extensions = (".jpg", ".jpeg", ".png")
        if not file_path.lower().endswith(allowed_extensions):
            return False
        return True
    
    @staticmethod
    def load_info(info_path):
        info = pd.read_csv(info_path, delimiter=' ')
        info['date'] = pd.to_datetime(info[['year', 'month', 'day']])
        info['SAMPLING_RATE[kHz]'] = info['SAMPLING_RATE[Hz]'].floordiv(1000)
        info.drop('SAMPLING_RATE[Hz]', axis=1, inplace=True)
        return info
    
    @staticmethod
    def extract_observ_data(path):
        template = r'(r|t)swf-e_(\d{4})(\d{2})(\d{2})'
        search = re.search(template, path)
        observation_type = search.group(1)
        date = datetime.date(int(search.group(2)), int(search.group(3)), int(search.group(4)))
        return observation_type, date
    
    def save_database(df, **kwargs):
        dir_name = './data/'
        try:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
        except:
            print(f'Can not create directory {dir_name}')

        kind = kwargs.get('kind', 'pickle')
        file_name = kwargs.get('file_name', 'pearson_diagram_data')
        try:
            if kind == 'pickle':
                df.to_pickle(dir_name+file_name+'.pkl')
            elif kind == 'json':
                df.to_json(dir_name+file_name+'.json')
        except:
            print('Can not save the datbase')

    def read_database(**kwargs):
        
        kind = kwargs.get('kind', 'pickle')
        file = kwargs.get('file', './data/pearson_diagram_data')
        print(f'Reading a database from a file {file}')
        try:
            if kind == 'pickle':
                df = pd.read_pickle(file+'.pkl')
                print(df.head())
                print(df.dtypes)
                return df
            elif kind == 'json':
                df = pd.read_json(file+'.json', dtype={'obsertype': 'category', 'label': 'category', 'date': 'datetime'})
                print(df.head())
                print(df.dtypes)
                return df
        except:
            print('Error during reading a database')
            return None