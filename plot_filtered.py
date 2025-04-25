from  soloinfo import *

import numpy as np
import pandas as pd
import seaborn as sns

df_reg_data = pd.read_pickle('./data/pearson_diagram_data.pkl')

df_reg_filtered = df_reg_data[df_reg_data['label'].isin([0,1])]

cat = solo_info('./data/pearson_diagram_data.pkl')

cat.data = df_reg_filtered

fields = ['dist_to_sun[au]','SAMPLES_NUMBER', 'SAMPLING_RATE[Hz]','SAMPLE_LENGTH[ms]']
#cat.print_stat(fields)
#print(cat.data)
#cat.plot_stat(fields)
axes = cat.plot_stat_one_ax(fields,figsize=(16,6))
axes['dist_to_sun[au]'].invert_yaxis()
axes['dist_to_sun[au]'].plot(df_reg_data['date'], df_reg_data['dist_to_sun[au]'], color='gray',ls='',marker='.', zorder=0)
print(axes['dist_to_sun[au]'])
plt.tight_layout()
plt.show()