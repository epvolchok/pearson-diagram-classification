import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
class solo_info(object):
    def __init__(self,file_path):
        self.file_path=file_path
        #print('hi')

    def read(self):
        # Загружаем данные из текстового файла
        self.data = pd.read_csv(self.file_path, delimiter=' ')  # Используйте delimiter='\t' для табуляции или delimiter=',' для запятых
        self.data['date'] = pd.to_datetime(self.data[['year', 'month', 'day']])


    def get(self,val,key='date'):
        print('get')
        filtered_df = self.data[self.data[key] == val]
        print(val,' ',filtered_df)
        return filtered_df    
  
    
    def print_stat(self,fields):
        # Выводим статистику для указанных полей
        statistics = self.data[fields].describe()
        print(statistics)

    def plot_stat(self,fields):

        # Создаем график
        plt.figure(figsize=(10, 6))
            # Создаем фигуру и оси
        fig, axes = plt.subplots(nrows=len(fields), ncols=1, figsize=(10, 6 * len(fields)), sharex=True)

        # Строим графики для каждого поля
        for ax, field in zip(axes, fields):
            sns.lineplot(data=self.data, x='date', y=field, ax=ax)
            ax.set_title(f'Распределение {field} от даты')
            ax.set_ylabel(field)

        # Добавляем метки осей для последнего графика
        axes[-1].set_xlabel('Дата')
        
    def plot_stat_one_ax(self,fields,figsize=(10,6)):

        # Создаем график
            # Создаем фигуру и оси
        fig, ax = plt.subplots(figsize=figsize)
        # Цвета для линий
        colors = plt.cm.tab10(range(len(fields)))
        axes = {}
        # Строим графики для каждого поля
        for i, (field, color) in enumerate(zip(fields, colors)):
            print(i,field)
            
            if i == 0:
                axes[field]=(ax)
                # Основная ось
                label0 = self.data[self.data['label'].isin([0])]
                label1 = self.data[self.data['label'].isin([1])]
                ax.plot(label0['date'], label0[field], label=field, color='blue', ls='',marker='.', zorder=1)
                ax.plot(label1['date'], label1[field], label=field, color='purple', ls='',marker='.', zorder=2)

                ax.set_ylabel(field, color=color)
                ax.tick_params(axis='y', labelcolor=color)
            else:
                # Дополнительная ось
                ax_new = ax.twinx()
                axes[field]=(ax_new)
                ax_new.spines['right'].set_position(('outward', 60 * i))  # Смещение оси
                ax_new.plot(self.data['date'], self.data[field], label=field, color=color,ls='',marker='.')
                ax_new.set_ylabel(field, color=color)
                ax_new.tick_params(axis='y', labelcolor=color)

        # Добавляем заголовок и метки осей
        #plt.title('Распределение полей от даты')
        ax.set_xlabel('Date')
        return axes
        # Показываем легенду
        #fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

    def find(self,key1,val1,key2,val2):
        # Значения, которые нужно найти
        value1_to_find = val1
        value2_to_find = val2
        df =self.data
        # Вычисляем евклидово расстояние для каждой строки
        df['distance'] = np.sqrt((df[key1] - value1_to_find) ** 2 + (df[key2] - value2_to_find) ** 2)

        # Находим строку с минимальным расстоянием
        closest_row = df.loc[df['distance'].idxmin()]
        # Поиск строки, где значения соответствуют заданным
        print(closest_row)
        return closest_row
