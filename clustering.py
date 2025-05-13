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
    name_pattern = r'(solo_L2_rpw-tds-surv-(?:r|t)swf-e_\d+\w+)'
    clear = ServiceFuncs.get_bool(f'If folder {results_dir} already exists would you like to clear its contents? (True or False) ')
    print(clear)
    ServiceFuncs.preparing_folder(results_dir, clear=clear)

    features = InteractiveMode.get_features(input_imags, default_info_path, default_filename, name_pattern) #ResNet object

    print('Done!')
    print(features.database.head())
    InteractiveMode.run_processing(features, default_info_path, default_filename, name_pattern)


if __name__ == '__main__':
    main()


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