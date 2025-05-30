import pandas as pd
import seaborn as sns
import sys

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

    def __init__(self, filename: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series, int]:
        self.df = DBFuncs.read_database(filename)
        self.df['year'] = self.df['date'].dt.year
        self.label_counts = self.df['label'].value_counts()
        self.num_clusters = self.label_counts.size
        return self.df, self.label_counts, self.num_clusters

    def hist_preparations(self):
        self.xs = ['dist_to_sun[au]', 'SAMPLES_NUMBER', 'SAMPLING_RATE[kHz]', 'SAMPLE_LENGTH[ms]', 'year']
        self.titles = ['Distance to Sun, au', 'Number of samples', 'Sampling rate, kHz', 'Sample length, ms', 'year']
        self.colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green']

    def plot_one_hist(ax, data, xs, color, bin_num, title, xrange, label):

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

    def plot_hists(fig, df, num_clusters):

        pass