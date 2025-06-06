import pandas as pd 
from manalyse import Plotter

import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

def main():
    Plt = Plotter('./results/pearson_diagram_data_reg_b_clustered')
    """
    cluster_labels = {
    (0, 0): r'\textbf{Type A}',
    (0, 1): r'\textbf{Type B}',
    (0, 2): r'\textbf{Mostly}'+'\n'+r'\textbf{Type A}',
    (0, 3): r'\textbf{Low}'+'\n'+r'\textbf{number}'+'\n'+r'\textbf{of samples }'}
    """
    print(Plt.labels)
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(Plt.num_clusters, len(Plt.fields))
    Plt.plot_hists(gs)
    plt.tight_layout()
    Plt.save_fig(filename='hist_clusters')

    fig, ax = plt.subplots(figsize=(16,9))
    point_colors = ['tab:blue', 'tab:red']
    Plt.plot_dist_to_sun(ax, labels_to_plot=Plt.labels[1:3], point_colors=point_colors)
    plt.tight_layout()
    Plt.save_fig(filename='dist_to_sun')


    plt.show()

if __name__ == '__main__':
    
    main()
