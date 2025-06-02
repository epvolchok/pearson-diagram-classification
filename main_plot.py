import pandas as pd 
from manalyse import HistPlotter
from mclustering import Logg

import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

if __name__ == '__main__':
    
    log_name = Logg.get_log_filename()
    Logg.setup_logging(log_name)
    
    Plt = HistPlotter('./results/pearson_diagram_data_reg_b_clustered')
    labels = Plt.label_counts_all
    df = Plt.df
    labels = labels.loc[~(labels == labels.loc[-1])]
    num = Plt.num_clusters
    print(num)

    cluster_labels = {
    (0, 0): r'\textbf{Type A}',
    (0, 1): r'\textbf{Type B}',
    (0, 2): r'\textbf{Mostly}'+'\n'+r'\textbf{Type A}',
    (0, 3): r'\textbf{Low}'+'\n'+r'\textbf{number}'+'\n'+r'\textbf{of samples }'}
    Plt.hist_preparations(cluster_labels=None)

    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(num, len(Plt.xs))
    print(labels)


    Plt.plot_hists(gs)
    plt.tight_layout()
    Plt.save_fig()
    plt.show()