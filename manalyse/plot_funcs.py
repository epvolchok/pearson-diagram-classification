import pandas as pd
import seaborn as sns
import numpy as np
import os

import matplotlib
matplotlib.use('qtagg')
from matplotlib import rc
import matplotlib.pyplot as plt

from typing import Optional, Tuple, List

rc('text', usetex=True)
rc('font', family='serif')
rc('text.latex', preamble=r"\usepackage[utf8]{inputenc}")
matplotlib.rcParams.update({'font.size': 14})


class Plotter:
    """
    Visualization of SoLo measurements.
    Reads data from a .pkl or .json database, builds histograms and scatter plot based on years, 
    depending on the distance from the sun and other measurement parameters.
    """

    def __init__(self, filename: Optional[str] = None) -> None:
        self.df = Plotter.read_database(filename)
        self.df['year'] = self.df['date'].dt.year

        self.label_counts_all = self.df['label'].value_counts()
        label_counts = self.label_counts_all.loc[~(self.label_counts_all == self.label_counts_all.loc[-1])]
        self.labels = label_counts.index.values
        self.counts_per_cluster = label_counts.values
        self.num_clusters = self.labels.size

        self.fields = ['dist_to_sun[au]', 'SAMPLES_NUMBER', 'SAMPLING_RATE[kHz]', 'SAMPLE_LENGTH[ms]', 'year']
        self.xytitles = ['Distance to Sun, au', 'Number of samples', 'Sampling rate, kHz', 'Sample length, ms', 'Year']

    @staticmethod 
    def read_database(file_to_read: Optional[str] = None, kind: str = 'pickle') -> Optional[pd.DataFrame]:
        """
        Read a pandas DataFrame from a file.

        Parameters
        ----------
        file_to_read : str, optional
            Base filename without extension.
        kind : {'pickle', 'json'}
            Format of the file.
        dtype : dict, optional
            Column types to enforce when reading JSON.

        Returns
        -------
        pandas.DataFrame or None
            The loaded DataFrame, or None if reading failed.
        """
        default_name = os.path.join(os.getcwd(), 'results', 'pearson_diagram_data')
        file_to_read = Plotter.create_name(default_name, file_to_write=file_to_read)
        
        try:
            if kind == 'pickle':
                df = pd.read_pickle(file_to_read+'.pkl')
            elif kind == 'json':
                df = pd.read_json(file_to_read+'.json')
            return df
        except Exception as e:
            print(f'Error during reading a database: {e}')
            return None

    @staticmethod
    def create_name(default_filename: str, file_to_write: Optional[str] = None, suffix: Optional[str] = None) -> str:
        """
        Constructs a filename by combining a base name with a suffix.

        Parameters
        ----------
        default_filename : str
            Base filename.
        file_to_write : str, optional
            Custom filename provided by the user.
        suffix : str, optional
            Suffix to append to the filename.

        Returns
        -------
        str
            Full filename with optional suffix.
        """
        if not file_to_write:
            file_to_write = default_filename
        if suffix:
            file_to_write += suffix
        return file_to_write

    def hist_preparations(self, cluster_labels: Optional[dict] = None,
                          colors: Optional[List[str]] = None,
                          bins: Optional[List[int]] = None) -> Tuple[dict, List, List[int]]:
        """
        Prepare histogram settings: labels, colors, bin sizes.

        Parameters
        ----------
        cluster_labels : dict, optional
            Dictionary of labels to display on each subplot.
        colors : list, optional
            List of colors for clusters.
        bins : list of int, optional
            Number of bins per field.

        Returns
        -------
        tuple
            (cluster_labels, colors, bins)
        """

        hist_cluster_labels = {} if cluster_labels is None else cluster_labels
        hist_colors = matplotlib.cm.tab10(range(self.num_clusters)) if colors is None else colors

        if bins is None:
            hist_bins = [10, 15]
            hist_bins.extend(list(np.full(len(self.fields)-2, 5, dtype=int)))
        else:
            hist_bins = bins
        
        return hist_cluster_labels, hist_colors, hist_bins

    @staticmethod
    def plot_one_hist(ax: plt.Axes, data: pd.DataFrame, xs: str, color: str,
                      bin_num: int, xtitle: str, xrange: Optional[Tuple[float, float]], label: str) -> None:
        """
        Plot a single histogram for one field and cluster.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to plot on.
        data : pandas.DataFrame
            Subset of the data to plot.
        xs : str
            Column name for x-axis.
        color : str
            Color of the histogram.
        bin_num : int
            Number of histogram bins.
        xtitle : str
            X-axis title.
        xrange : tuple or None
            X-axis limits.
        label : str
            Y-axis label.
        """

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

    def plot_hists(self, gs, cluster_labels: Optional[dict] = None,
                   colors: Optional[List[str]] = None,
                   bins: Optional[List[int]] = None) -> None:
        """
        Plot histograms of selected fields for each cluster using a grid layout.

        Parameters
        ----------
        gs : matplotlib.gridspec.GridSpec
            GridSpec layout.
        cluster_labels : dict, optional
            Text annotations per subplot.
        colors : list of colors, optional
            Colors for each cluster.
        bins : list of int, optional
            Number of bins for each field.
        """
        cluster_labels, colors, bins = self.hist_preparations(cluster_labels, colors, bins)
        for iy in range(self.num_clusters):
            cluster_labels[(0, iy)] =cluster_labels.get((0, iy), '') + \
            '\n' + r'\textbf{num = '+str(self.counts_per_cluster[iy])+' }'
            for ix in range(len(self.fields)):
                ax = plt.subplot(gs[iy, ix])
                df = self.df[self.df['label'] == self.labels[iy]]
                bin_num = bins[ix]
                xtitle = ''
                xrange = None
                label = ''

                if (ix, iy) in cluster_labels:
                    label = cluster_labels[(ix, iy)]
                if iy == 0:
                    xtitle = self.xytitles[ix]

                Plotter.plot_one_hist(ax=ax, data=df, xs=self.fields[ix], color=colors[iy], \
                                bin_num=bin_num, xtitle=xtitle, xrange=xrange, label=label)
                

    def plot_preparations(self, labels_to_plot: Optional[np.ndarray] = None,
                          labels: Optional[List[str]] = None,
                          axes_colors: Optional[List[str]] = None,
                          point_colors: Optional[List[str]] = None) -> Tuple[List[str], List[str], np.ndarray, List[str]]:
        """
        Prepare color palettes and label strings for plotting by year.

        Parameters
        ----------
        labels_to_plot : array-like, optional
            Labels to include.
        labels : list of str, optional
            Custom label names.
        axes_colors : list, optional
            Colors for each y-axis.
        point_colors : list, optional
            Colors for scatter points.

        Returns
        -------
        tuple
            (axes_colors, point_colors, labels_to_plot, print_label)
        """
        default_colors = plt.cm.Dark2(range(len(self.fields)+1)) 
        default_colors = np.delete(default_colors, 1, axis=0)
        axes_colors = default_colors if axes_colors is None else axes_colors

        
        labels_to_plot = self.labels if labels_to_plot is None else labels_to_plot
        print_label = ['label ' + str(l) for l in labels_to_plot] if labels is None else labels
        point_colors = matplotlib.cm.tab10(range(self.num_clusters)) if point_colors is None else point_colors

        return axes_colors, point_colors, labels_to_plot, print_label
    
    def plot(self, ax: plt.Axes, field: str, labels_to_plot: np.ndarray,
             print_label: List[str], point_colors: List[str]) -> None:
        """
        Plot field values over time for each cluster.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to plot on.
        field : str
            Field to plot.
        labels_to_plot : list or array
            Cluster labels to include.
        print_label : list of str
            Label names to use in legend.
        point_colors : list of str
            Colors for each cluster.
        """
        for j, label in enumerate(labels_to_plot):
            df_label = self.df[self.df['label']==label][['date', field]]
            ax.plot(df_label['date'], df_label[field], label=print_label[j], \
                    color=point_colors[j], ls='', marker='.', ms=10, zorder=j+1)
        ax.plot(self.df['date'], self.df[field], label='', color='gray', lw=2, zorder=0)

    def plot_twin_ax(self, ax_new: plt.Axes, i: int, field: str, color: str) -> None:
        """
        Plot a field on a new twin y-axis.

        Parameters
        ----------
        ax_new : matplotlib.axes.Axes
            Twin axis.
        i : int
            Index of the field.
        field : str
            Field to plot.
        color : str
            Color of the plot.
        """
        ax_new.spines['right'].set_position(('outward', 60 * (i-1)))
        ax_new.plot(self.df['date'], self.df[field], label=field, color=color,ls='',marker='.')
        ax_new.set_ylabel(r'\textbf{'+self.xytitles[i]+'}', color=color, fontsize=16)
        ax_new.tick_params(axis='y', labelcolor=color)

    def plot_dist_to_sun(self, ax: plt.Axes, labels_to_plot: Optional[np.ndarray] = None,
                    labels: Optional[List[str]] = None, axes_colors: Optional[List[str]] = None,
                    point_colors: Optional[List[str]] = None) -> None:
        """
        Plot distance to sun and other metadata fields as time series
        using shared x-axis and multiple y-axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Main axis.
        labels_to_plot : list, optional
            Cluster labels.
        labels : list of str, optional
            Custom label names.
        axes_colors : list, optional
            Colors for y-axes.
        point_colors : list, optional
            Colors for points.
        """
        axes_colors, point_colors, labels_to_plot, print_label = self.plot_preparations(labels_to_plot, labels, axes_colors, point_colors)
        axes = {}
        for i, (field, color) in enumerate(zip(self.fields, axes_colors)):
            if i == 0:
                axes[field]=(ax)
                self.plot(ax, field, labels_to_plot, print_label, point_colors)
                ax.set_ylabel(r'\textbf{'+self.xytitles[i]+'}', color=color, fontsize=16)
                ax.tick_params(axis='y', labelcolor=color)
            else:
                ax_new = ax.twinx()
                axes[field]=(ax_new)
                self.plot_twin_ax(ax_new, i, field, color)
        ax.set_xlabel(r'\textbf{Date}', fontsize=16)
        axes['dist_to_sun[au]'].legend(loc='lower left', bbox_to_anchor=(0.07,0.2)) #, bbox_to_anchor=(0.07,1)


                
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
        if filename is None:
            filename = default_name
        file_png = os.path.join(fig_dir, filename+'.png')
        file_pdf = os.path.join(fig_dir, filename+'.pdf')
        plt.savefig(file_pdf, format='pdf')
        plt.savefig(file_png, format='png', dpi=300)