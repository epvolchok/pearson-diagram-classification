import os
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import umap
from torchvision import models, transforms
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import hdbscan
import re
import datetime
import pandas as pd

class Clustering:

    def __init__(self, path, device='cuda'):
        self._device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f'The device is {self._device}')
        self.names = [f for f in os.listdir(path) if Clustering.check_extension(f)]
        self.img_path = [path + f for f in os.listdir(path) if Clustering.check_extension(f)]
        self.features = []

    @staticmethod
    def check_extension(file_path):
        allowed_extensions = (".jpg", ".jpeg", ".png")
        if not file_path.lower().endswith(allowed_extensions):
            return False
        return True
    
    def model_resnet50(self):
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
    
    @staticmethod
    def models_dict():
        models_dict = {
            'pca': PCA(n_components=0.95, svd_solver='full'),
            'scaler': StandardScaler(),
            'umap10d': umap.UMAP(n_components=10, min_dist=0.1, metric='cosine'),
            'umap2d': umap.UMAP(n_components=2)
        }
        return models_dict

    def preproccess(self, pipe_str): #pipe_str='PCA+UMAP+..'
        models = pipe_str.lower().split('+')
        pipes = []
        for m in models:
            pipes.append((m, Clustering.models_dict()[m]))
        if self.features.size:
            pipe = Pipeline(pipes)
            results = pipe.fit_transform(self.features)
            return results
        else:
            return None
    
    def feature_extraction(self):
        
        model = self.model_resnet50()[0].eval()
        transform = self.model_resnet50()[1]
        
        for path in tqdm(self.img_path, desc='Extracting features'):
            try:
                img = Image.open(path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    feat = model(img_tensor).squeeze().cpu().numpy()
                self.features.append(feat)
            except:
                print(f'Error while openning an image {self.img_path}')
        
        #self.features = np.array(self.features)

    def write_features(self, path='./'):
        if self.features:
            try:
                np.save(path + 'features.npy', self.features)
            except:
                print('Error during writing data')
        else:
            print('Extract features first')
    
    def read_features(self, path='./features.npy'):
        try:
            data = np.load(path)
            self.features = data
            return self.features
        except:
            print('Error during reading data')
            return None
    
    @staticmethod
    def extract_observ_data(path):
        template = r'(r|t)swf-e_(\d{4})(\d{2})(\d{2})'
        search = re.search(template, path)
        observation_type = search.group(1)
        date = datetime.date(int(search.group(2)), int(search.group(3)), int(search.group(4)))
        return observation_type, date
    
    def create_database(self, labels):
        num = len(labels)
        types = []
        dates = []
        for name in self.names:
            type_, date_ = Clustering.extract_observ_data(name)
            types.append(type_)
            dates.append(date_)
        df = pd.DataFrame({
            'filename': self.names,
            'observation type': types,
            'date': dates,
            'labels': labels,
            'path': self.img_path
        })
        print(df.head())
        return df
    def create_dirs(df):
        if not os.path.exists('./processed'):
            os.makedirs('my_folder')
        


    def sort_files(self, df):
        for i, row in enumerate(df):
            if i < 10:
                print(row)
            else:
                break
        