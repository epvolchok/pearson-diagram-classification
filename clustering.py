#Copyright (c) 2025 Evgeniia VOLCHOK
#for contacts e.p.volchok@gmail.com

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0


import matplotlib.pyplot as plt
import os
import re

from libpreprocessing import FeaturesPreprocessing
from libclustering import Clustering
from libfeatures import ResNetFeatures
from libservice import ServiceFuncs
from libinteractive import InteractiveMode

def main():

    base_dir_names = ['./images', './processed', './figures', './data', './results']
    for dirname in base_dir_names:
        ServiceFuncs.preparing_folder(dirname, clear=False)

    InteractiveMode.welcome_message()
    input_imags = input('Please enter the name of your working directory: ').strip('./, ')
    print(input_imags)


    specification = ServiceFuncs.input_name(input_imags)
    print(specification)

    results_dir = './processed/processed_'+specification
    default_filename = './results/pearson_diagram_data_'+specification
    default_info_path = './data/SOLO_info_rswf.txt'
    name_pattern = r'(solo_L2_rpw-tds-surv-rswf-e_\d+\w+)'
    clear = ServiceFuncs.get_bool(f'If folder {results_dir} already exists would you like to clear its contents? (True or False) ')
    print(clear)
    ServiceFuncs.preparing_folder(results_dir, clear=clear)

    features = InteractiveMode.get_features(input_imags, default_info_path, default_filename, name_pattern)

    print('Done!')
    print(features.database.head())


if __name__ == '__main__':
    main()

""" print('Features extraction')
features = ResNetFeatures(input_imags, flag='extract', file_to_write='./data/pearson_diagram_data_triggered')
print('Filtration by variance')
features.database = features.filtering_by_variance()
features.info_on_features()
print('Saving database')
ServiceFuncs.save_database(features.database, file_to_write='filtered_pearson_diagram_data_triggered')
print('Preprocessing')
preprop = FeaturesPreprocessing(features)
processed = preprop.wrapper_preprop(features.database, 'PCA+UMAP10D')
df_features, excluded_part = ServiceFuncs.split_into_two(processed)
print('Clusterization')
clusters = Clustering(processed)
labels, num_clusters = clusters.clustering_HDBSCAN(df_features)
print(f'Number of clusters 20D: {num_clusters}')
clusters.update_database()
print('Saving database')
ServiceFuncs.save_database(clusters.df, file_to_write='clustered_pearson_diagram_data_triggered', kind='json')
clusters.sort_files()
clusters.visualize_HDBSCAN(features.database) """



""" obj = Clustering(input_imags)

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

""" 
"""plt.figure(figsize=(7,5))

plt.subplot(1, 2, 1)
plt.hist(variances_after, bins=50, color='skyblue')
plt.title('Variance')
plt.xlabel('Variance')
plt.ylabel('Frequency') """

"""from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=5e-5)
varfiltered_features = selector.fit_transform(filtered_features)
variances_after_thershold = np.var(varfiltered_features, axis=0)

""" 
"""print("Осталось признаков:", varfiltered_features.shape[1])
plt.subplot(1, 2, 2)
plt.hist(variances_after_thershold, bins=50, color='skyblue')
plt.title('Variance')
plt.xlabel('Variance')
plt.ylabel('Frequency') """



"""
print(np.std(varfiltered_features, axis=0))
stds = np.std(varfiltered_features, axis=0)
nonzero = [el for el in stds if el != 0]
print(f'non zero {len(nonzero)}')
#print(nonzero)
print(f'std {np.std(nonzero)}')
distances = pairwise_distances(varfiltered_features, metric='cosine')
mean_dist = distances[np.triu_indices_from(distances, k=1)].mean()

print('Average cosine distance between embeddings:', mean_dist)

""" 
"""filtered_var_features = filtered_features[:, (variances_after > 0)]
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
plt.grid(True) """
""" plt.show()

# 5. Сравним количественно
print(f"Было признаков: {features.shape[1]}")
print(f"Осталось после фильтрации: {filtered_features.shape[1]}")
print(f"Осталось после фильтрации: {varfiltered_features.shape[1]}")
print(f"Средняя дисперсия до: {np.mean(variances_before):.5f}")
print(f"Средняя дисперсия после: {np.mean(variances_after):.5f}")
print(f"Средняя дисперсия после: {np.mean(varfiltered_features):.5f}")

obj.features = varfiltered_features

features_processed = obj.preproccessing('PCA+UMAP10D')

labels, n_clusters = obj.clustering_HDBSCAN(features_processed)
print(f'Found {n_clusters} clusters (10D)')

df = obj.database
obj.sort_files()
obj.save_database()

obj.read_database()
obj.visualize() """