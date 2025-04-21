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
from sklearn.metrics import pairwise_distances
from libcluster import Clustering

input_imags = './images/'


obj = Clustering(input_imags)

#features = obj.feature_extraction()
#obj.write_features()

features = obj.read_features()
print(np.std(features, axis=0))
stds = np.std(features, axis=0)
nonzero = [el for el in stds if el != 0]
print(f'non zero {len(nonzero)}')
#print(nonzero)
print(f'std {np.std(nonzero)}')
distances = pairwise_distances(features, metric='cosine')
mean_dist = distances[np.triu_indices_from(distances, k=1)].mean()

print("Средняя косинусная дистанция между эмбеддингами:", mean_dist)


features_processed = obj.preproccess('PCA+UMAP10D')

labels, n_clusters = obj.clustering_HDBSCAN(features_processed)
print(f"Найдено кластеров: {n_clusters}")

df = obj.database
obj.sort_files()

"""
# 🎨 Визуализация
plt.figure(figsize=(10, 8))
palette = plt.get_cmap("tab10")

for label in set(labels):
    mask = labels == label
    color = 'gray' if label == -1 else palette(label % 10)
    plt.scatter(umap_2d[mask, 0], umap_2d[mask, 1], s=10, color=color, label=f'Кластер {label}' if label != -1 else 'Шум')

plt.legend()
plt.title("UMAP + HDBSCAN: визуализация кластеров")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.show()
"""
