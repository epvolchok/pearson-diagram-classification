#Copyright (c) 2025 Evgeniia VOLCHOK
#for contacts e.p.volchok@gmail.com

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0


import pandas as pd
import numpy as np
import os
from PIL import Image
import re
import datetime
import subprocess



class ServiceFuncs:

    def __init__(self):
        raise RuntimeError('This class can not be instantiate.')
    
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
    
    @staticmethod
    def save_database(df, **kwargs):
        dir_name = './results/'
        try:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
        except:
            print(f'Can not create directory {dir_name}')

        kind = kwargs.get('kind', 'pickle')
        file_to_write = kwargs.get('file_to_write', 'pearson_diagram_data')
        print(f'Saving the database to {file_to_write}')
        try:
            if kind == 'pickle':
                df.to_pickle(file_to_write+'.pkl')
            elif kind == 'json':
                df.to_json(file_to_write+'.json')
        except:
            print('Can not save the database')

    @staticmethod
    def read_database(**kwargs):
        
        kind = kwargs.get('kind', 'pickle')
        file_to_read = kwargs.get('file_to_read', 'pearson_diagram_data')
        print(f'Reading a database from a file {file_to_read}')
        try:
            if kind == 'pickle':
                df = pd.read_pickle(file_to_read+'.pkl')
                print(df.head())
                print(df.dtypes)
                return df
            elif kind == 'json':
                df = pd.read_json(file_to_read+'.json', dtype={'obsertype': 'category', 'label': 'category', 'date': 'datetime'})
                print(df.head())
                print(df.dtypes)
                return df
        except:
            print('Error during reading a database')
            return None
        
    @staticmethod
    def preparing_folder(dir_name, clear):
        
        if os.path.exists(dir_name):
            if clear:
                try:
                    subprocess.run(['rm', '-r', dir_name], check=True)
                    os.makedirs(dir_name)
                except subprocess.CalledProcessError as e:
                    print(f'Error during deleting files in {dir_name}: {e}')
        else:
            os.makedirs(dir_name)

    @staticmethod
    def split_into_two(df, excluded_columns=['dataset_name', 'date', 'dist_to_sun[au]', 'SAMPLES_NUMBER', 'SAMPLING_RATE[kHz]',
                    'SAMPLE_LENGTH[ms]', 'oldpath']):
        mask = [cols for cols in df.columns if cols not in excluded_columns]
        df_features = df.loc[:, mask]
        excluded_part = df.loc[:, excluded_columns]
        return df_features, excluded_part

    @staticmethod
    def input_name(input_imags):
        pattern = r'images_(\w+)'
        search = re.search(pattern, input_imags)
        if search:
            return search.group(1)
        else:
            return input_imags
        
    @staticmethod
    def get_bool(prompt):
        while True:
            try:
                return {'true': True, 'false': False}[input(prompt).lower()]
            except KeyError:
                print('Invalid input, please enter True or False!')
