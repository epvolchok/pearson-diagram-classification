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
import subprocess


import matplotlib
matplotlib.use('qtagg')
from matplotlib import rc

rc('text', usetex=True)
rc('font', family='serif')
rc('text.latex', preamble=r"\usepackage[utf8]{inputenc}")
matplotlib.rcParams.update({'font.size': 14})

from functools import cached_property

class Clustering:

    def __init__(self, path, info_path='./data/SOLO_info_rswf.txt', device='cuda'):
        
        self._device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f'The device is {self._device}')
        self.names = [f for f in os.listdir(path) if Clustering.check_extension(f)]
        self.img_path = [path]*len(self.names)
        self.img_path = [p + self.names[i] for i, p in enumerate(self.img_path)]

        self.info = Clustering.load_info(info_path)
        self.filtering_imgs()

        self.num_clusters = 0
        self.labels = []

        if not os.path.exists('./processed'):
            os.makedirs('./processed')

        self.models_dict = {
            'pca': PCA(n_components=0.95, svd_solver='full'),
            'scaler': StandardScaler(),
            'umap10d': umap.UMAP(n_components=10, min_dist=0.3, metric='cosine'),
            'umap2d': umap.UMAP(n_components=2, min_dist=0.3, metric='cosine')
        }

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
        return info
    
    def find_mixed_freq(self):
        high_freq = 524
        low_freq = 262

        tgt = self.info['SAMPLING_RATE[Hz]'].floordiv(1000)
        mask = ~(
            ((tgt >= low_freq - 5) & (tgt <= low_freq + 5)) |
            ((tgt >= high_freq - 5) & (tgt <= high_freq + 5))
        )
        
        df = self.info[mask]['dataset_name']
        return df
    

    def filtering_imgs(self):
        name_pattern = r'(solo_L2_rpw-tds-surv-rswf-e_\d+\w+)'
        
        mixed_freq = self.find_mixed_freq()
        
        for i, path in enumerate(self.img_path):
            dataset_name = re.search(name_pattern, path).group(0)
            if dataset_name in mixed_freq.values:
                self.names.remove(dataset_name+'.png')
                self.img_path.remove('./images/'+dataset_name+'.png')



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

    def preproccessing(self, pipe_str): #pipe_str='PCA+UMAP+..'
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
        return np.array(features)

    def write_features(self, path='./data/'):
        if self.features.size:
            try:
                np.save(path + 'features.npy', self.features)
            except:
                print('Error during writing data')
        else:
            print('Extract features first')
    
    def read_features(self, path='./data/features.npy'):
        try:
            data = np.load(path)
            self.features = data
            return self.features
        except:
            print('Error during reading data')
            return None
        
    def clustering_HDBSCAN(self, features_processed):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5, metric='euclidean')
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
        self.names = [name[:-4] for name in self.names]
        df = pd.DataFrame({
            'dataset_name': self.names,
            'obsertype': types,
            'date': dates,
            'label': self.labels,
            'oldpath': self.img_path
        })
        df = df.astype({'obsertype': 'category', 'label': 'category', 'date': 'datetime64[ns]'})
        df_full = pd.merge(df, \
            self.info[['dataset_name', 'dist_to_sun[au]', 'SAMPLES_NUMBER', 'SAMPLING_RATE[Hz]', 'SAMPLE_LENGTH[ms]']], \
            how='left', on='dataset_name')

        return df_full
    

    def create_dirs(self):
        
        for cl in range(self.num_clusters):
            dir_name = './processed/label_'+str(cl)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
        if not os.path.exists('./processed/noise'):
            os.makedirs('./processed/noise')


    def create_newpath(self):
        for cl in range(self.num_clusters):
            self.database.loc[self.database['label'] == cl, 'path'] = './processed/label_'+str(cl)
        self.database.loc[self.database['label'] == -1, 'path'] = './processed/noise'



    def copy_files(self):
        for index, row in self.database.iterrows():
            oldpath = row['oldpath']
            newpath = row['path']
            if isinstance(oldpath, str) and isinstance(newpath, str):
                try:
                    subprocess.run(['cp', '-u', row['oldpath'], row['path']], check=True)
                except subprocess.CalledProcessError as e:
                    print(f'Error during copying {row['oldpath']} → {row['path']}: {e}')
            else:
                print(f'Probably missing path {index}: oldpath={oldpath}, path={newpath}')
        
    def sort_files(self):
        self.create_dirs()
        self.create_newpath()
        self.copy_files()
        self.database.drop('oldpath', axis=1, inplace=True)


    def save_database(self, kind='pickle'):
        dir_name = './data/'
        if not os.path.exists(dir_name):
                os.makedirs(dir_name)
        if kind == 'pickle':
            self.database.to_pickle(dir_name+'pearson_diagram_data.pkl')
        elif kind == 'json':
            self.database.to_json(dir_name+'pearson_diagram_data.json')

    def read_database(self, file='./data/pearson_diagram_data', kind='pickle'):

        try:
            if kind == 'pickle':
                self.database = pd.read_pickle(file+'.pkl')
                print(self.database.head())
                print(self.database.dtypes)
            elif kind == 'json':
                self.database = pd.read_json(file+'.json', dtype={'obsertype': 'category', 'label': 'category', 'date': 'datetime'})
                print(self.database.head())
                print(self.database.dtypes)
        except:
            print('Error during reading a database')

    def visualize(self):
        features_processed = self.preproccessing('PCA+UMAP2D')
        labels, n_clusters = self.clustering_HDBSCAN(features_processed)
        print(f'Found {n_clusters} clusters (2D)')
        plt.figure(figsize=(10, 8))
        palette = plt.get_cmap('tab10')

        for label in set(labels):
            mask = labels == label
            color = 'gray' if label == -1 else palette(label % 10)
            plt.scatter(features_processed[mask, 0], features_processed[mask, 1], s=10, color=color, label=f'Cluster {label}' if label != -1 else 'Noise')

        plt.legend()
        plt.title('PCA + UMAP2D + HDBSCAN: clustering visualization')
        plt.xlabel('UMAP-1 component')
        plt.ylabel('UMAP-2  component')
        plt.show()
        