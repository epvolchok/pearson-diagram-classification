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