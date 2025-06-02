import pandas as pd
import seaborn as sns
import sys
import numpy as np
import os

from mclustering import DBFuncs

import matplotlib
matplotlib.use('qtagg')
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib import gridspec
from typing import Optional, Tuple

rc('text', usetex=True)
rc('font', family='serif')
rc('text.latex', preamble=r"\usepackage[utf8]{inputenc}")
matplotlib.rcParams.update({'font.size': 14})

class HistPlotter:

    def __init__(self, filename: Optional[str] = None):
        self.df = DBFuncs.read_database(filename)
        self.df['year'] = self.df['date'].dt.year

        self.label_counts_all = self.df['label'].value_counts()
        label_counts = self.label_counts_all.loc[~(self.label_counts_all == self.label_counts_all.loc[-1])]
        self.labels = label_counts.index.values
        self.counts = label_counts.values
        self.num_clusters = self.labels.size



    def hist_preparations(self, cluster_labels=None, colors=None, bins=None): 
        self.xs = ['dist_to_sun[au]', 'SAMPLES_NUMBER', 'SAMPLING_RATE[kHz]', 'SAMPLE_LENGTH[ms]', 'year']
        self.xtitles = ['Distance to Sun, au', 'Number of samples', 'Sampling rate, kHz', 'Sample length, ms', 'year']

        self.cluster_labels = {} if cluster_labels is None else cluster_labels
        self.colors = matplotlib.cm.tab10(range(self.num_clusters)) if colors is None else colors

        if bins is None:
            self.bins = [10, 15]
            self.bins.extend(list(np.full(len(self.xs)-2, 5, dtype=int)))
        else:
            self.bins = bins

    @staticmethod
    def plot_one_hist(ax, data, xs, color, bin_num, xtitle, xrange, label):

        sns.histplot(data=data, x=xs, ax=ax, bins=bin_num, color=color)
        ax.set_xlabel(' ')
        ax.set_ylabel(' ')
        #print(label)
        if xrange:
            ax.set_xlim(xrange)
        if xtitle:
            ax.set_title(xtitle)
        if label:
            ax.set_ylabel(label, rotation=0, color=color, labelpad=40.)

    def plot_hists(self, gs):
    
        cluster_labels = self.cluster_labels
        for iy in range(self.num_clusters):
            cluster_labels[(0, iy)] =cluster_labels.get((0, iy), '') + '\n' + r'\textbf{num = '+str(self.counts[iy])+' }'#cluster_labels[(0, iy)] + '\n' + r'\textbf{num = '+str(nums[labels[iy]])+' }'
            for ix in range(len(self.xs)):
                ax = plt.subplot(gs[iy, ix])
                df = self.df[self.df['label'] == self.labels[iy]]
                bin_num = self.bins[ix]
                xtitle = ''
                xrange = None
                label = ''

                if (ix, iy) in cluster_labels:
                    label = cluster_labels[(ix, iy)]
                #if ix == 2 and (iy == 1 or iy == 2):
                    #xrange = (255, 269)
                    #bin_num = 1
                if iy == 0:
                    xtitle = self.xtitles[ix]

                HistPlotter.plot_one_hist(ax=ax, data=df, xs=self.xs[ix], color=self.colors[iy], \
                                bin_num=bin_num, xtitle=xtitle, xrange=xrange, label=label)
                
    def save_fig(self, filename: Optional[str] = None) -> None:
        """
        Saves the clustering plot in PDF and PNG formats.

        Parameters
        ----------
        filename : str
            Base filename to use for saving.
        """
        fig_dir = os.path.join(os.getcwd(), 'figures')
        default_name = 'cluster_hists'
        if not filename:
            filename = default_name
        file_png = os.path.join(fig_dir, filename+'.png')
        file_pdf = os.path.join(fig_dir, filename+'.pdf')
        plt.savefig(file_pdf, format='pdf')
        plt.savefig(file_png, format='png', dpi=300)