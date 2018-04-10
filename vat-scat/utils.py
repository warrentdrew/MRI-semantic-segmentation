'''
module for preprocessing
The data are stored in the .mat format 
Each patient folder contains a file named "rework.mat" which contains:
- img: MR image
- P_BG: background mask/label
- P_LT: lean tissue mask/label
- P_AT: (subcutaneous) adipose tissue mask/label
- P_VAT: visceral adipose tissue mask/label
- info: MR DICOM acquisition parameter and patient info
'''

import scipy.io as sio
import numpy as np
import time
import os
import pickle
import random
from libs.preprocessing import augmentation

file_path = '/med_data/Segmentation/AT/1_5T/PLIS_3609_GK/rework.mat'

mat = sio.loadmat(file_path)
print('img shape:', mat['img'].dtype)
print('SCAT shape:', mat['P_AT'].shape)
print('VAT shape:', mat['P_VAT'].shape)
print('bg shape:', mat['P_BG'].shape)

'''
1. store_patients: read in all the path of image(and mask) into a list, stored in pickle/h5
2. spliting the training and testing data path, split training and validation data path
3. for training data, next_batch() will read in the img of a new batch
4. for training labels, next_batch() will read in a mask containing all the 4 masks, merging them together
5. 
'''
patient_path_1_5T = '/med_data/Segmentation/AT/1_5T/'
#patient_list_1_5T = []

def store_patients(data_path, save_path = '../patient-paths/patients_1_5T.pkl'):
    #load patient paths from dataset
    patient_list = []
    for root, dirs, files in os.walk(data_path):
        for names in filter(lambda name: name[0] != '.', files):
            patient_list.append(os.path.join(root, names))

    #write patient paths into pickle
    # save paths on hard disk with pickle
    with open(save_path, 'wb+') as pathfile:
        pickle.dump(patient_list, pathfile)


def load_patient_paths(load_path = '../patient-paths/patients_1_5T.pkl'):
    #load all the paths into a list from the pickle file
    with open(load_path, 'rb') as pathfile:
        patient_list = pickle.load(pathfile)

    return patient_list

def dataset_split(load_path, test_rate = 0.2, valid_train_rate = 0.1, shuffle = True, seed = None):
    '''
    split training(+validation)/testing or training/validation data
    :param load_path:
    :param rate: 
    :param shuffle: 
    :param seed: 
    :return: train_path, validation_path, test_path
    '''
    patient_list = load_patient_paths(load_path)
    if shuffle:
        if seed is not None:
            random.seed(seed)
            random.shuffle(patient_list)
        else:
            random.shuffle(patient_list)

    test_patient_num = int(test_rate * len(patient_list))
    print('test_patient_num', test_patient_num)
    test_path = [patient_list[i] for i in range(test_patient_num)]
    validation_patient_num = int(valid_train_rate * (len(patient_list) -test_patient_num))
    print('validation_patient_num', validation_patient_num)
    validation_path = [patient_list[test_patient_num + i] for i in range(validation_patient_num)]
    train_patient_num = len(patient_list) - test_patient_num - validation_patient_num
    print('train_patient_num', train_patient_num)
    train_path = [patient_list[validation_patient_num + test_patient_num + i] for i in range(train_patient_num)]

    return train_path, validation_path, test_path

def load_data(path):
    '''
    load .mat data from path
    :param path:  
    :return: dictionary which contains the data and labels
    '''
    return sio.loadmat(path)

def merge_labels(path):
    '''
    merge 4 labels for each data set in following sequence
    1. bg : background
    2. lt : lean tissue
    3. vat: VAT
    4. scat: SCAT
    :param path:  
    :return: 
    '''

    patient_dict = load_data(path)
    bg = patient_dict['P_BG']
    lt = patient_dict['P_LT']
    vat = patient_dict['P_VAT']
    scat = patient_dict['P_AT']
    #label_mat = np.ndarray(bg.shape + (1,))
    new_shape = bg.shape + (1,)
    bg = bg.reshape(new_shape)
    lt = lt.reshape(new_shape)
    vat = vat.reshape(new_shape)
    scat = scat.reshape(new_shape)
    label_mat = bg
    #print('shape1:', label_mat.shape)
    label_mat = np.concatenate((label_mat, lt), axis = 3)
    label_mat = np.concatenate((label_mat, vat), axis = 3)
    label_mat = np.concatenate((label_mat, scat), axis = 3)
    #print('shape2:', label_mat.shape)

    return label_mat


def crop_data(img_mat, ):
    df


def get_batch(batch_size, train_path, item_index, stride, crop_per_patient, cropsize, num_data):
    num_patients_to_load = batch_size // crop_per_patient
    train_batch = np.ndarray(batch_size + 3 *(cropsize,))
    if
    for i in range(item_index, (item_index + num_patients_to_load) % num_data):
        patient_dict = load_data(train_path[i])
        patient_patches = crop_data(patient_dict['img'],cropsize,stride)
        train_batch.append(patient_patches)

    return train_batch










pkl_path = '../patient-paths/patients_1_5T.pkl'

#store_patients(patient_path_1_5T)
#load_patients()
#train_path, validation_path, test_path = dataset_split(pkl_path, shuffle= False)
#for i in range(len(train_path)):
#    print(train_path[i])


merge_labels(file_path)
