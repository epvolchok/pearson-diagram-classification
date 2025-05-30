import pandas as pd 
import manalyse as Plt
from mclustering import Logg

import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

if __name__ == '__main__':
    
    log_name = Logg.get_log_filename()
    Logg.setup_logging(log_name)
    df, label_counts, num_clusters = Plt.load_data('./results/pearson_diagram_data_reg_b_clustered')
    

    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(3, 5)
    print(label_counts)