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

from functools import cached_property

class Clustering:

    def __init__(self, path, device='cuda'):
        
        self._device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f'The device is {self._device}')
        self.names = [f for f in os.listdir(path) if Clustering.check_extension(f)]
        self.img_path = [path]*len(self.names)
        self.img_path = [p + self.names[i] for i, p in enumerate(self.img_path)]

        self.num_clusters = 0
        self.labels = []

        if not os.path.exists('./processed'):
            os.makedirs('./processed')

        self.models_dict = {
            'pca': PCA(n_components=0.95, svd_solver='full'),
            'scaler': StandardScaler(),
            'umap10d': umap.UMAP(n_components=10, min_dist=0.1, metric='cosine'),
            'umap2d': umap.UMAP(n_components=2)
        }

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

    """
     @staticmethod
    def models_dict():
        models_dict = {
            'pca': PCA(n_components=0.95, svd_solver='full'),
            'scaler': StandardScaler(),
            'umap10d': umap.UMAP(n_components=10, min_dist=0.1, metric='cosine'),
            'umap2d': umap.UMAP(n_components=2)
        }
        return models_dict
    """

    def preproccess(self, pipe_str): #pipe_str='PCA+UMAP+..'
        models = pipe_str.lower().split('+')
        pipes = []
        for m in models:
            pipes.append((m, self.models_dict[m]))
        if self.features.size:
            pipe = Pipeline(pipes)
            results = pipe.fit_transform(self.features)
            return results
        else:
            return None
    
    @cached_property
    def features(self):
        
        model = self.model_resnet50()[0].eval()
        transform = self.model_resnet50()[1]
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
        return features

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
        
    def clustering_HDBSCAN(self, features_processed):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=15, metric='euclidean')
        self.labels = clusterer.fit_predict(features_processed)
        self.num_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        return self.labels, self.num_clusters
    
    @staticmethod
    def extract_observ_data(path):
        template = r'(r|t)swf-e_(\d{4})(\d{2})(\d{2})'
        search = re.search(template, path)
        observation_type = search.group(1)
        date = datetime.date(int(search.group(2)), int(search.group(3)), int(search.group(4)))
        return observation_type, date
    
    @cached_property
    def database(self):
        num = len(self.labels)
        types = []
        dates = []
        for name in self.names:
            type_, date_ = Clustering.extract_observ_data(name)
            types.append(type_)
            dates.append(date_)
        print(f'dates {len(dates)}, types {len(types)}, labels {len(self.labels)}')
        df = pd.DataFrame({
            'filename': self.names,
            'observation type': types,
            'date': dates,
            'label': self.labels,
            'oldpath': self.img_path
        })
        #print(df.head())
        return df
    

    def create_dirs(self):
        
        for cl in range(self.num_clusters):
            dir_name = './processed/label_'+str(cl)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)


    def create_newpath(self):
        for cl in range(self.num_clusters):
            self.database.loc[self.database['label'] == cl, 'path'] = './processed/label_'+str(cl)
        print(self.database.head())


    def copy_files(self):
        for index, row in self.database.iterrows():
            #print('cp '+row['oldpath']+' '+row['path'])
            os.system('cp '+row['oldpath']+' '+row['path'])
        
    def sort_files(self):
        self.create_dirs()
        self.create_newpath()
        self.copy_files()

    def visualize(self):
        features_processed = self.preproccess('PCA+UMAP10D')
        labels, n_clusters = self.clustering_HDBSCAN(features_processed)

        plt.figure(figsize=(10, 8))
        palette = plt.get_cmap('tab10')

        for label in set(labels):
            mask = labels == label
            color = 'gray' if label == -1 else palette(label % 10)
            plt.scatter(features_processed[mask, 0], features_processed[mask, 1], s=10, color=color, label=f'Cluster {label}' if label != -1 else 'Noise')

        plt.legend()
        plt.title('UMAP + HDBSCAN: clustering visualization')
        plt.xlabel('UMAP-1 component')
        plt.ylabel('UMAP-2  component')
        plt.show()
        