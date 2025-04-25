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

input_imags = './images_reg_b/'


obj = Clustering(input_imags)

#features = obj.features
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

print('Average cosine distance between embeddings:', mean_dist)

# 1. Дисперсия до фильтрации
variances_before = np.var(features, axis=0)

# 2. Удаляем признаки, которые равны 0 во всех строках
non_zero_columns = ~(features == 0).all(axis=0)
filtered_features = features[:, non_zero_columns]

# 3. Дисперсия после фильтрации
variances_after = np.var(filtered_features, axis=0)

plt.figure(figsize=(7,5))

plt.subplot(1, 2, 1)
plt.hist(variances_after, bins=50, color='skyblue')
plt.title('Variance')
plt.xlabel('Variance')
plt.ylabel('Frequency')

from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=5e-5)
thresholdfiltered_features = selector.fit_transform(filtered_features)
variances_after_thershold = np.var(thresholdfiltered_features, axis=0)

print("Осталось признаков:", thresholdfiltered_features.shape[1])
plt.subplot(1, 2, 2)
plt.hist(variances_after_thershold, bins=50, color='skyblue')
plt.title('Variance')
plt.xlabel('Variance')
plt.ylabel('Frequency')




print(np.std(thresholdfiltered_features, axis=0))
stds = np.std(thresholdfiltered_features, axis=0)
nonzero = [el for el in stds if el != 0]
print(f'non zero {len(nonzero)}')
#print(nonzero)
print(f'std {np.std(nonzero)}')
distances = pairwise_distances(thresholdfiltered_features, metric='cosine')
mean_dist = distances[np.triu_indices_from(distances, k=1)].mean()

print('Average cosine distance between embeddings:', mean_dist)

filtered_var_features = filtered_features[:, (variances_after > 0)]
variances_after_after = np.var(filtered_var_features, axis=0)

thresholds = np.linspace(0, 0.01, 10000)
n_features_remaining = []

for t in thresholds:
    mask = variances_after > t
    n_features_remaining.append(np.sum(mask))

percentiles = [0.1, 0.5, 1, 2, 5, 7, 10]  # процентили, которые хотим проверить

for p in percentiles:
    threshold = np.percentile(variances_after, p)
    num_features = np.sum(variances_after > threshold)
    print(f"{p}%-percentile threshold = {threshold:.8f}, remaining features: {num_features}")



# 3. Строим график
plt.figure(figsize=(8,5))
plt.plot(thresholds, n_features_remaining, marker='o', markersize=3)
percentile_thresholds = [np.percentile(variances_after, p) for p in percentiles]
for p, t in zip(percentiles, percentile_thresholds):
    plt.axvline(x=t, linestyle='--', label=f'{p}%-порог = {t:.6f}', alpha=0.7, color='red')
plt.title('Threshold influence on features')
plt.xlabel('Variance threshold')
plt.ylabel('Remaining reatures')
plt.legend()
plt.grid(True)
plt.show()

# 5. Сравним количественно
print(f"Было признаков: {features.shape[1]}")
print(f"Осталось после фильтрации: {filtered_features.shape[1]}")
print(f"Осталось после фильтрации: {filtered_var_features.shape[1]}")
print(f"Средняя дисперсия до: {np.mean(variances_before):.5f}")
print(f"Средняя дисперсия после: {np.mean(variances_after):.5f}")
print(f"Средняя дисперсия после: {np.mean(variances_after_after):.5f}")

""" features_processed = obj.preproccessing('PCA+UMAP10D')

labels, n_clusters = obj.clustering_HDBSCAN(features_processed)
print(f'Found {n_clusters} clusters (10D)')

df = obj.database
obj.sort_files()
obj.save_database()

obj.read_database()
obj.visualize()  """