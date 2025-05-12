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


df_reg_data = pd.read_json('./data/clustered_pearson_diagram_data_triggered.json')
print(df_reg_data.head())
print(df_reg_data.info())
print('\n')
print(df_reg_data.describe(include='all'))


print(df_reg_data['label'].value_counts())

df_reg_data['year'] = df_reg_data['date'].dt.year

path = './processed/label_'
labels = df_reg_data['label'].value_counts().index.values
#print(labels)
xs = ['dist_to_sun[au]', 'SAMPLES_NUMBER', 'SAMPLING_RATE[kHz]', 'SAMPLE_LENGTH[ms]', 'year']
titles = ['Distance to Sun, au', 'Number of samples', 'Sampling rate, kHz', 'Sample length, ms', 'year']
colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green']
fig = plt.figure(figsize=(16, 9))
gs = gridspec.GridSpec(3, 5)

label = 0
iy = 0
ix = 2

def plot_hist(ax, data, xs, color, bin_num, title, xrange, label):

    sns.histplot(data=data, x=xs, ax=ax, bins=bin_num, color=color)
    ax.set_xlabel(' ')
    ax.set_ylabel(' ')
    #print(label)
    if xrange:
        ax.set_xlim(xrange)
    if title:
        ax.set_title(title)
    if label:
        ax.set_ylabel(label, rotation=0, color=color, labelpad=40.)

bins = [10, 10, 5, 5, 5]

cluster_labels = {
    (0, 0): r'\textbf{Type A}',
    (0, 1): r'\textbf{Type B}',
    (0, 2): r'\textbf{Mostly}'+'\n'+r'\textbf{Type A}',
    (0, 3): r'\textbf{Low}'+'\n'+r'\textbf{number}'+'\n'+r'\textbf{of samples }'}

nums = df_reg_data['label'].value_counts()
#print(nums[3])
#for iy in range(len(labels)):
#    cluster_labels[(0, iy)]


for iy in range(len(labels)):
    cluster_labels[(0, iy)] = r'\textbf{num = '+str(nums[labels[iy]])+' }'#cluster_labels[(0, iy)] + '\n' + r'\textbf{num = '+str(nums[labels[iy]])+' }'
    for ix in range(len(xs)):
        ax = plt.subplot(gs[iy, ix])
        df = df_reg_data[df_reg_data['label'] == labels[iy]]
        bin_num = bins[ix]
        title = ''
        xrange = None
        label = ''

        if (ix, iy) in cluster_labels:
            label = cluster_labels[(ix, iy)]
        #if ix == 2 and (iy == 1 or iy == 2):
            #xrange = (255, 269)
            #bin_num = 1
        if iy == 0:
            title = titles[ix]

        plot_hist(ax, df, xs[ix], colors[iy], bin_num, title, xrange, label)
plt.tight_layout()
plt.savefig('./figures/histograms_triggered.pdf', format='pdf', dpi=300)


plt.show()
