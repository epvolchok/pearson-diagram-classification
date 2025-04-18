import os
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
from torchvision import models, transforms
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import hdbscan
import json

class Clustering:

    def __init__(self, path, device='cuda'):
        self._device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f'The device is {self._device}')
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
            return data
        except:
            print('Error during reading data')
            return None