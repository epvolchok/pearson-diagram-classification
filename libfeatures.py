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

from libservice import ServiceFuncs

class ResNetFeatures:
    def __init__(self, path, info_path='./data/SOLO_info_rswf.txt', device='cuda'):

        self._device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f'The device is {self._device}')

        self.names = [f for f in os.listdir(path) if ServiceFuncs.check_extension(f)]
        self.img_path = [path]*len(self.names)
        self.img_path = [p + self.names[i] for i, p in enumerate(self.img_path)]
        self.info = ServiceFuncs.load_info(info_path)
        self.filtering_imgs(path)

        self.database = self.create_database()

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
    

    def filtering_imgs(self, path):
        name_pattern = r'(solo_L2_rpw-tds-surv-rswf-e_\d+\w+)'
        
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
            except:
                print(f'Error while openning an image {self.img_path}')
        return np.array(features)
    
    
    def create_database(self):
        features = self.features()
        df_features = pd.DataFrame(
            features,
            columns=[f"feat_{i}" for i in range(features.shape[1])]
        )
        
        self.names = [name[:-4] for name in self.names]
        df_features.insert(0, 'oldpath', self.img_path)
        df_features.insert(0, 'dataset_name', self.names)
        
        

        #df = df.astype({'obsertype': 'category', 'label': 'category', 'date': 'datetime64[ns]'})
        df_full = pd.merge(
            self.info[['dataset_name', 'date', 'dist_to_sun[au]', 'SAMPLES_NUMBER', 'SAMPLING_RATE[kHz]', 'SAMPLE_LENGTH[ms]']], 
            df_features, 
            how='left', on='dataset_name')
        print(df_full.info())
        print(df_full.head())
        return df_full
