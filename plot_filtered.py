import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif')
rc('text.latex', preamble=r"\usepackage[utf8]{inputenc}")
matplotlib.rcParams.update({'font.size': 16})

df_full = pd.read_csv('./data/SOLO_info_tswf.txt', delimiter=' ')
df_full['date'] = pd.to_datetime(df_full[['year', 'month', 'day']])

df_reg_data = pd.read_json('./data/clustered_pearson_diagram_data_triggered.json')

df_reg_filtered = df_reg_data[df_reg_data['label'].isin([0,1,2,3])]


fields = ['dist_to_sun[au]','SAMPLES_NUMBER', 'SAMPLING_RATE[kHz]','SAMPLE_LENGTH[ms]']
titles = ['Distance to the Sun, au', 'Number of samples', 'Sampling rate, kHz', 'Sample length, ms', 'Year']

fig, ax = plt.subplots(figsize=(16,9))
# Цвета для линий
colors = plt.cm.Dark2(range(len(fields)+1))
print(colors)
colors = np.delete(colors, 1, axis=0)
axes = {}
# Строим графики для каждого поля
for i, (field, color) in enumerate(zip(fields, colors)):
    print(i,field)
    
    if i == 0:
        axes[field]=(ax)
        # Основная ось

        label0 = df_reg_data[df_reg_data['label'].isin([0])]
        label1 = df_reg_data[df_reg_data['label'].isin([1])]
        #label2 = df_reg_data[df_reg_data['label'].isin([3])]
        label2 = df_reg_data[df_reg_data['label'].isin([2])]
        ax.plot(label1['date'], label1[field], label='Type B?', color='tab:orange', ls='',marker='.', ms=10,  zorder=2)
        ax.plot(label2['date'], label2[field], label='Type As?', color='tab:red', ls='',marker='.', ms=10,  zorder=5)
        ax.plot(label0['date'], label0[field], label='Type ?', color='tab:blue', ls='',marker='.', ms=10, zorder=1)
        #ax.plot(label3['date'], label3[field], label='Low number of samples', color='tab:green', ls='',marker='.', ms=10, zorder=1)

        ax.plot(df_full['date'], df_full[field], label='', color='gray', lw=2, zorder=0)

        ax.set_ylabel(r'\textbf{'+titles[i]+'}', color=color, fontsize=16)
        ax.tick_params(axis='y', labelcolor=color)
    else:
        # Дополнительная ось
        ax_new = ax.twinx()
        axes[field]=(ax_new)
        ax_new.spines['right'].set_position(('outward', 60 * (i-1)))  # Смещение оси
        ax_new.plot(df_reg_data['date'], df_reg_data[field], label=field, color=color,ls='',marker='.')
        ax_new.set_ylabel(r'\textbf{'+titles[i]+'}', color=color, fontsize=16)
        ax_new.tick_params(axis='y', labelcolor=color)

# Добавляем заголовок и метки осей
#plt.title('Распределение полей от даты')
ax.set_xlabel(r'\textbf{Date}', fontsize=16)

axes['dist_to_sun[au]'].invert_yaxis()
axes['dist_to_sun[au]'].legend(loc='upper left', bbox_to_anchor=(0.07,1))
#axes['dist_to_sun[au]'].plot(df_reg_data['date'], df_reg_data['dist_to_sun[au]'], color='gray',ls='',marker='.', zorder=0)
print(axes['dist_to_sun[au]'])
plt.tight_layout()

plt.savefig('./figures/Dist_to_Sun_trig.pdf', format='pdf', dpi=300)
plt.savefig('./figures/Dist_to_Sun_trig.png', format='png', dpi=300)
plt.show()