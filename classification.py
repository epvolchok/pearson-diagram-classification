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

input_imags = './images/'


#img = Image.open(input_imags+'solo_L2_rpw-tds-surv-rswf-e_20210101_V03.png').convert('RGB')
#Extracting features
img_path = [input_imags + f for f in os.listdir(input_imags) if f[-3:] == 'png']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224)),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225])
])

model_resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model_resnet50 = torch.nn.Sequential(*list(model_resnet50.children())[:-1])  # without the last layer
model_resnet50.to(device)
model_resnet50.eval()


features = []
for path in tqdm(img_path, desc='Extracting features'):
    try:
        img = Image.open(path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model_resnet50(img_tensor).squeeze().cpu().numpy()
        features.append(feat)
    except:
        print(f'Error while openning an image {img_path}')
    

features = np.array(features)

#standardtirize before compressing dimenssions
features_scaled = StandardScaler().fit_transform(features)

# reducing components

model_pca = PCA(n_components=25, svd_solver='randomized', random_state=42)
features_PCA = model_pca.fit_transform(features_scaled)

model_umap = umap.UMAP(n_components=10, min_dist=0.1, metric='cosine', random_state=42)
features_umap = model_umap.fit_transform(features_PCA)

# clustering
model_hdbscan = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, metric='euclidean', cluster_selection_method='eom')
labels = model_hdbscan.fit_predict(features_umap)
print(labels)
#plt.scatter(umap_features[:, 0], umap_features[:, 1], s=10)
#plt.title("UMAP проекция")
#plt.show()