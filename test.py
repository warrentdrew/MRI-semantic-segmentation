from libs.util import check_patients

path = '/home/d1251/med_data/KORA/'

lst_channel = ['fat0', 'water1']
lst_class = ['liver', 'spleen']

check_patients(path, lst_channel, lst_class, path_save = 'pickle-data/', verbose = True)
