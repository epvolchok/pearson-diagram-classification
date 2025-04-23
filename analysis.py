import os
from PIL import Image
import numpy as np
import pandas as pd
import seaborn as sns
import random

import matplotlib
matplotlib.use('qtagg')
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib import gridspec

rc('text', usetex=True)
rc('font', family='serif')
rc('text.latex', preamble=r"\usepackage[utf8]{inputenc}")
matplotlib.rcParams.update({'font.size': 14})


from libcluster import Clustering

df_reg_data = pd.read_pickle('./data/pearson_diagram_data.pkl')
print(df_reg_data.head())
print(df_reg_data.info())
print('\n')
print(df_reg_data.describe(include='all'))


print(df_reg_data['label'].value_counts())
#sns.histplot(data=df_r, x='dist_to_sun[au]', hue='label', element='step')

df_reg_data['SAMPLING_RATE[kHz]'] = df_reg_data['SAMPLING_RATE[Hz]'].floordiv(1000)
df_reg_data.drop('SAMPLING_RATE[Hz]',  axis=1, inplace=True)

path = './processed/label_'
labels = [0, 2, 1]
xs = ['dist_to_sun[au]', 'SAMPLES_NUMBER', 'SAMPLING_RATE[kHz]', 'SAMPLE_LENGTH[ms]']
colors = ['blue', 'orange', 'green']
fig = plt.figure(figsize=(20, 12))
gs = gridspec.GridSpec(3, 4)
for iy in range(len(labels)):
    for ix in range(len(xs)):
        ax = plt.subplot(gs[iy, ix])
        df = df_reg_data[df_reg_data['label'] == labels[iy]]
        sns.histplot(data=df, x=xs[ix], ax=ax, bins=5, color=colors[iy])
        ax.set_xlabel(' ')
        ax.set_ylabel(' ')
        if ix == 0:
            ax.set_ylabel(r'\textbf{label '+str(labels[iy])+'}', rotation=0, color=colors[iy], labelpad=30.)
        if iy == 0:
            ax.set_title(xs[ix])
#plt.tight_layout()

for iy in range(len(labels)):
    fig2 = plt.figure(figsize=(10, 5))
    gs2 = gridspec.GridSpec(1, 1)
    ax2 = plt.subplot(gs2[0, 0])
    path_ = path+str(labels[iy])+'/'
    files = os.listdir(path_)
    img = Image.open(path_+random.choice(files)).convert('RGB')
    ax2.imshow(img, aspect='auto')
    ax2.axis("off")
    ax2.set_title(r'\textbf{label '+str(labels[iy])+'}', rotation=0, color=colors[iy])
    plt.tight_layout()


"""
fig = plt.figure(figsize=(15, 4))
gs = gridspec.GridSpec(1, 5)
for iy in range(5):
    ax = plt.subplot(gs[0, iy])
    df = df_r[df_r['label'] == labels[iy]]
    sns.histplot(data=df, x=xs[1], ax=ax)
    ax.set_title(labels[iy])
    ax.set_xlabel(xs[1]) """

""" fig = plt.figure(figsize=(15, 4))
gs = gridspec.GridSpec(1, 5)
for iy in range(5):
    ax = plt.subplot(gs[0, iy])
    df = df_r[df_r['label'] == labels[iy]]
    sns.histplot(data=df, x=xs[2], ax=ax)
    ax.set_title(labels[iy])
    ax.set_xlabel(xs[2]) """

""" fig = plt.figure(figsize=(15, 4))
gs = gridspec.GridSpec(1, 5)
for iy in range(5):
    ax = plt.subplot(gs[0, iy])
    df = df_r[df_r['label'] == labels[iy]]
    sns.histplot(data=df, x=xs[3], ax=ax)
    ax.set_title(labels[iy])
    ax.set_xlabel(xs[3]) """

plt.tight_layout()
plt.show()
