from  soloinfo import *

cat = solo_info('./SOLO_info_rswf.txt')
cat.read()
one_day = cat.get('2020-11-25')
one_day = cat.get('solo_L2_rpw-tds-surv-rswf-e_20200618_V05',key='dataset_name')


# Выводим статистику для указанных полей
fields = ['dist_to_sun[au]','SAMPLES_NUMBER', 'SAMPLING_RATE[Hz]','SAMPLE_LENGTH[ms]']
#cat.print_stat(fields)
#print(cat.data)
#cat.plot_stat(fields)
axes = cat.plot_stat_one_ax(fields,figsize=(16,6))
axes['dist_to_sun[au]'].invert_yaxis()
print(axes['dist_to_sun[au]'])
plt.tight_layout()
plt.show()


# Значения, которые нужно найти
value1_to_find = 30
value2_to_find = 25

#cat.find('SAMPLING_RATE[Hz]',262100,'SAMPLE_LENGTH[ms]',125)