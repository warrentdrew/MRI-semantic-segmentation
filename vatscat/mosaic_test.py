import matplotlib
matplotlib.use('TkAgg')

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from patient import Patient_AT
#from keras.models import load_model
#import libs.custom_metrics as custom_metrics
#import blocks
#from dccn_symm import DCCN_SYMM
#from mrge_net import MRGE
from utils import load_patient_paths

import tensorflow as tf
import keras.backend as K
import random

patient_list = load_patient_paths('../patient-paths/patients_1_5T.pkl', shuffle=True, seed= 100)
patient_path = random.choice(patient_list)

dataset_rt = "/Users/zhuyipin/Documents/AT/AT"

head1, tail1 = os.path.split(patient_path)
head2, tail2 = os.path.split(head1)
_, tail3 = os.path.split(head2)
id = os.path.join(os.path.join(tail3, tail2), tail1)

patient_local_path = os.path.join(dataset_rt, id)
patient_local_path = '/Users/zhuyipin/Documents/AT/AT/1_5T/PLIS2_3529_TL/rework.mat'
print('patient local path:', patient_local_path)

patient = Patient_AT(patient_path= patient_local_path, forget_slices = True)
patient.save_mosaic_style(col = 5, space = 1, back = 1)
#plt1, fig = patient.check_slice(depth= 50)
#fig.savefig("../img-test/test.png")
#plt1.show()

