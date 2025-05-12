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
        message = 'Welcome to the script! \n \
        First place your images in a separate folder named according to the template: \n \
            "images_(your_specification)" \n \
        into "./images/" folder.'
        print(message)

    @staticmethod
    def filter_mixed_freq(name_pattern):
        filter_mixed = ServiceFuncs.get_bool('Would you like to filter mixed frequencies? (True or False) ')
        if filter_mixed:
            print(f'Check the name pattern: {name_pattern}')
            np_input = input('Change if it is needed or press "Enter": ').strip()
            if np_input:
                name_pattern = r'+np_input+'
        return filter_mixed, name_pattern
    
    @staticmethod
    def get_path(message, default_path):
        file_path = input(message)
        file_path = file_path if file_path else default_path
        return file_path
    
    @staticmethod
    def get_features(input_imags, default_info_path, default_filename, name_pattern):
        print(f'To extract features from images located in {input_imags} enter 1.')
        print(f'To load features from a file enter 2.')
        while True:
            choice = int(input('[1/2]: ').strip())
            
            message = f'Enter a path to the data description file (default "{default_info_path}"): '
            info_path = InteractiveMode.get_path(message, default_info_path)

            if choice == 1:
                message = f'Enter a file name to write extracted features (default "{default_filename}"): '
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
                break

            elif choice == 2:
                
                message = f'Enter a file name to read from (default "{default_filename}"): '
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
                break
            else:
                print('Invalid input. \n Try again!')
        return features

