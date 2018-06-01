import numpy as np

import os
from vatscat.utils import data_clean
from vatscat.utils import store_patients
data1 = '/home/d1251/no_backup/d1251/DATASET/AT/3T/'
data2 = '/med_data/Segmentation/AT/'
#
# store_patients(data_path = data1, save_path = './patient-paths/patients_3T.pkl')

x = np.arange(-2, 2)
y = np.arange(0, 3)
z = np.arange(1, 5)

a, b, c = np.meshgrid(x,y,z)

print('x:', x)
print('y:', y)
print('z:', z)
print('a:', a)
print('b:', b)
print('c:', c)

