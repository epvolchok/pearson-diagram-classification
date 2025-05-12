#Copyright (c) 2025 Evgeniia VOLCHOK
#for contacts e.p.volchok@gmail.com

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0


from torchvision import models, transforms
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from PIL import Image
import re
from functools import cached_property
import datetime
import matplotlib.pyplot as plt

from libservice import ServiceFuncs
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import pairwise_distances

class ResNetFeatures:

    def __init__(self, path, **kwargs): #path, info_path='./data/SOLO_info_rswf.txt', device='cuda'): filter_mixed

        device = kwargs.pop('device', 'cuda')
        info_path = kwargs.pop('info_path', './data/SOLO_info_rswf.txt')
        flag = kwargs.pop('flag', 'read')
        filter_mixed = kwargs.pop('filter_mixed', True)

        self.device = device
        self.info_path = info_path
        self.path = path


        self._device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f'The device is {self._device}')

        self.names = [f for f in os.listdir(path) if ServiceFuncs.check_extension(f)]
        self.img_path = [path]*len(self.names)
        self.img_path = [p + self.names[i] for i, p in enumerate(self.img_path)]
        self.info = ServiceFuncs.load_info(info_path)
        if filter_mixed:
            name_pattern = kwargs.pop('name_pattern', r'(solo_L2_rpw-tds-surv-rswf-e_\d+\w+)')
            self.filtering_imgs(path, name_pattern)

        if flag == 'read':
            self.database = ServiceFuncs.read_database(**kwargs)
        elif flag == 'extract':
            self.database = self.create_database()
            ServiceFuncs.save_database(self.database)
        else:
            raise ValueError(f'Unknown flag: {flag}')


    @cached_property
    def model(self):
        transform = transforms.Compose([
        transforms.Resize((224)),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])
                                    ])

        model_resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model_resnet50 = torch.nn.Sequential(*list(model_resnet50.children())[:-1])  # without the last layer
        model_resnet50.to(self._device)

        return model_resnet50, transform

    def find_mixed_freq(self):
        high_freq = 524
        low_freq = 262

        tgt = self.info['SAMPLING_RATE[kHz]']
        mask = ~(
            ((tgt >= low_freq - 5) & (tgt <= low_freq + 5)) |
            ((tgt >= high_freq - 5) & (tgt <= high_freq + 5))
        )
        
        df = self.info[mask]['dataset_name']
        return df
    

    def filtering_imgs(self, path, name_pattern=r'(solo_L2_rpw-tds-surv-rswf-e_\d+\w+)'):
        
        mixed_freq = self.find_mixed_freq()
        
        for p in self.img_path:
            dataset_name = re.search(name_pattern, p).group(0)
            if dataset_name in mixed_freq.values:
                self.names.remove(dataset_name+'.png')
                self.img_path.remove(path+dataset_name+'.png')

    def features(self):
        
        model, transform = self.model
        model.eval()
        features = []
        for path in tqdm(self.img_path, desc='Extracting features'):
            try:
                img = Image.open(path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    feat = model(img_tensor).squeeze().cpu().numpy()
                features.append(feat)
            except OSError as err:
                print(f'Error while openning an image {self.img_path}: {err}')
        return np.array(features)
    
    
    def create_database(self):
        features = self.features()
        try:
            df_features = pd.DataFrame(
                features,
                columns=[f'feat_{i}' for i in range(features.shape[1])]
            )

            self.names = [name[:-4] for name in self.names]
            df_features.insert(0, 'oldpath', self.img_path)
            df_features.insert(0, 'dataset_name', self.names)
            
            
            df_full = pd.merge(
                self.info[['dataset_name', 'date', 'dist_to_sun[au]', 'SAMPLES_NUMBER', 'SAMPLING_RATE[kHz]', 'SAMPLE_LENGTH[ms]']], 
                df_features, 
                how='left', on='dataset_name')
            df_full.dropna(inplace=True, ignore_index=True)
            print('Database with information about observation parameters and extracted features:')
            print(df_full.info())
            print(df_full.head())
            return df_full
        except Exception as err:
            print(f'Error during the feature extraction or database creation: {err}')
        
        

    def filtering_nonzerocolumns(self):
        
        df_features, excluded_part = ServiceFuncs.split_into_two(self.database)
        non_zero_columns = ~(df_features == 0).all(axis=0)
        filtered_features = df_features.loc[:, non_zero_columns]
        final_df = pd.concat([excluded_part, filtered_features], axis=1)
        print('Filtration of zero columns. Remaining size:')
        print(final_df.shape[1])
        return final_df
    
    def filtering_by_variance(self, threshold=5e-5):
        df_features, excluded_part = ServiceFuncs.split_into_two(self.database)
        selector = VarianceThreshold(threshold=threshold)
        filtered_features = selector.fit_transform(df_features)
        df_filtered = pd.DataFrame(
            filtered_features,
            columns=[f'feat_{i}' for i in range(filtered_features.shape[1])]
        )
        print('Filtration by the variance threshold. Remaining size:')
        final_df = pd.concat([excluded_part, df_filtered], axis=1)
        print(final_df.shape[1])
        return final_df
    
    def info_on_features(self, **kwargs):
        df_features, _ = ServiceFuncs.split_into_two(self.database)

        distances = pairwise_distances(df_features, metric='cosine')
        mean_dist = distances[np.triu_indices_from(distances, k=1)].mean()

        print('Average cosine distance between embeddings:', mean_dist)
        variances = np.var(df_features, axis=0)
        print(f'Average variance: {np.mean(variances)}')

        if kwargs.get('vis', False) == True:

            plt.figure(figsize=(7,5))
            plt.subplot(1, 2, 1)
            plt.hist(variances, bins=50, color='skyblue')
            plt.ylabel('Frequency')
            if 'title' in kwargs:
                plt.title(kwargs['title'])
            else:
                plt.title('Variance')


