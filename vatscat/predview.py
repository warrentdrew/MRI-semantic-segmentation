import matplotlib
matplotlib.use('TkAgg')

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from patient import Patient_AT
from keras.models import load_model
import libs.custom_metrics as custom_metrics
import blocks
from dccn_symm import DCCN_SYMM
from mrge_net import MRGE
from utils import dataset_split, load_patient_paths

import tensorflow as tf
import keras.backend as K
import random

# _, _ , test_path = dataset_split('../patient-paths/patients_3T.pkl', test_rate=0.2, valid_train_rate=0.05, shuffle=True, seed= 100)

patient_paths = load_patient_paths(load_path = '../patient-paths/patients_1_5T.pkl', shuffle = True, seed = 100)

model_dir = '/home/d1251/no_backup/d1251/models/'
#dccn_model_name = 'k-fold-symm-weights-0.hdf5'
mrge_model_name = 'k-fold-mrge-weights-pos0.hdf5'
#dccn_model_path = os.path.join(model_dir,dccn_model_name)
mrge_model_path = os.path.join(model_dir,mrge_model_name)

'''

dccn = DCCN_SYMM(in_shape=(32, 32, 32, 1),
                    kls = [16, 16, 32, 64],
                    ls=[8, 8, 4, 2],
                    theta=0.5,
                    k_0=32)
'''

mrge_pos = MRGE(in_shape = (32,32,32,1), rls = [8,4,2,1,1], k_0 = 16, feed_pos = True)


K.get_session().run(tf.global_variables_initializer())
#dccn.model.load_weights(dccn_model_path)
mrge_pos.model.load_weights(mrge_model_path)

#custom_object_list = get_custom_object_dict()
#model = load_model(filepath=model_path, custom_objects= custom_object_list)
'''

for i in range(5):

    patient_path = random.choice(test_path)
    print('patient path:', patient_path)
    # patient_path = '/home/d1251/no_backup/d1251/DATASET/AT/1_5T/PLIS_3609_GK/rework.mat'
    patient = Patient_AT(patient_path, forget_slices=True)
    depth = random.randint(35, 75)
    print('depth:', depth)
    fig1 = patient.plot_result(depth=depth, model=dccn.model, model_name='dccn_sy')
    fig1.draw()
    fig2 = patient.plot_result(depth=depth, model=mrge.model, model_name='mrge')
    fig2.draw()
fig1.show()
fig2.show()

'''
for i in range(10):
    patient_path = random.choice(patient_paths[:100])
    patient = Patient_AT(patient_path, forget_slices=True)
    patient.save_mosaic_style(model = mrge_pos.model, model_name = 'mrge-pos', col = 5, border = 20, space = 1, back = 1, save_rt = '/home/d1251/no_backup/d1251/results/mrge-pos/prediction/1_5T')

