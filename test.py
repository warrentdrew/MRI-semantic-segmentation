from libs.util import check_patients
import numpy as np
'''
path = '/home/d1251/med_data/KORA/'

lst_channel = ['fat0', 'water1']
lst_class = ['liver', 'spleen']

check_patients(path, lst_channel, lst_class, path_save = 'pickle-data/', verbose = True)
'''

a = np.ndarray()

b = np.zeros([1,3])
c = np.ones([1,3])

res = np.concatenate((a,b), axis = 0)
print(res)