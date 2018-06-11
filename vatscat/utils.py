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
#from preprocess import augmentation
from skimage.morphology import closing, cube


'''
file_path = '/med_data/Segmentation/AT/1_5T/PLIS_3609_GK/rework.mat'

mat = sio.loadmat(file_path)
print('img shape:', mat['img'].dtype)
print('SCAT shape:', mat['P_AT'].shape)
print('VAT shape:', mat['P_VAT'].shape)
print('bg shape:', mat['P_BG'].shape)
'''
#the shape of the patient matrix is 192, 256, 105

'''
1. store_patients: read in all the path of image(and mask) into a list, stored in pickle/h5
2. spliting the training and testing data path, split training and validation data path
3. for training data, next_batch() will read in the img of a new batch
4. for training labels, next_batch() will read in a mask containing all the 4 masks, merging them together
'''
patient_path_1_5T = '/med_data/Segmentation/AT/1_5T/'
#patient_list_1_5T = []
batch_num = 0

def check_patient(patient_path_list):
    for patient in patient_path_list:
        try:
            patient_data_shape = load_data(patient)['img'].shape
            patient_bg_shape = load_data(patient)['P_BG'].shape
            patient_lt_shape = load_data(patient)['P_LT'].shape
            patient_vat_shape = load_data(patient)['P_VAT'].shape
            patient_scat_shape = load_data(patient)['P_AT'].shape
            if not patient_data_shape[-1] >= 90:
                print('remove patient {} , shape is {},{},{},{},{}'.format(patient, patient_data_shape, patient_bg_shape, patient_lt_shape, patient_vat_shape, patient_scat_shape))
                patient_path_list.remove(patient)
        except ValueError as err:
            print('skip patient due to value error:', err)
            patient_path_list.remove(patient)
            continue

    return patient_path_list


def retrieve_pixel_classes(cls_vector):
    #print('cls vector shape:', cls_vector.shape)
    res = np.where(cls_vector == 1)[0]
    return res

def val_to_one_hot_array(value, class_num):
    res = np.zeros([class_num,])
    res[value] = 1
    return res


def label_remove_2_classes(label_mat):
    class_map = np.zeros(label_mat.shape[:3])
    zero_pixel_pos_list = []
    for x , row in enumerate(label_mat):
        for y, col in enumerate(row):
            for z, prd in enumerate(col):
                res = retrieve_pixel_classes(prd)
                if len(res) == 2:                   #len(res) shows how many classes the pixel belongs to
                    newres = np.delete(res, np.where(res == 3))
                    if len(newres) == 1:
                        class_map[x, y, z] = newres[0] + 1
                    else:
                        print('still more than 1 class!')
                elif  len(res) == 0:
                    class_map[x,y,z] = 0     #0 class pixels
                    zero_pixel_pos_list.append((x,y,z))

                elif len(res) == 1:
                    class_map[x,y,z] = res[0] + 1  #1 class pixels
                else:
                    print('pixel {},{},{} has more than 2 classes'.format(x,y,z))

    return class_map , zero_pixel_pos_list


def class_map_zero_pixel_closing(class_map, zero_pixel_pos_list,range):                #class map is the added up masks with no class pixels as 0

    #print('list length:', len(zero_pixel_pos_list))
    increase_range = True
    if len(zero_pixel_pos_list) == 0:
        #print('sdsda:')
        return class_map
    else:
        clean_map = class_map
        class_map = closing(class_map, cube(range)).astype(np.int8)
        for (x,y,z) in zero_pixel_pos_list:
            #print('haha:', (x, y, z))
            clean_map[x,y,z] = class_map[x,y,z]
            if class_map[x,y,z] != 0:
                zero_pixel_pos_list.remove((x,y,z))
                increase_range = False
        if increase_range:
            range += 1

        return class_map_zero_pixel_closing(clean_map, zero_pixel_pos_list,range)

def retrieve_label_from_class_map(class_map):
    label_map = np.zeros(class_map.shape + (4,))
    for x, row in enumerate(class_map):
        for y, col in enumerate(row):
            for z, prd in enumerate(col):
                #print('map value:', class_map[x, y, z])
                label_map[x,y,z,:] = val_to_one_hot_array((class_map[x,y,z]-1), 4)

    return label_map

def process_label(label_mat):
    double_class_removed_map, zero_pixel_pos_list = label_remove_2_classes(label_mat)
    class_map = class_map_zero_pixel_closing(double_class_removed_map, zero_pixel_pos_list, 2).astype(np.int8)
    clean_label = retrieve_label_from_class_map(class_map)
    return clean_label




def store_patients(data_path, save_path = '../patient-paths/patients_1_5T.pkl'):
    #load patient paths from dataset
    patient_list = []
    for root, dirs, files in os.walk(data_path):
        for names in filter(lambda name: name[0] != '.', files):
            print('name test:', os.path.join(root, names))
            patient_list.append(os.path.join(root, names))

    print('number of all patients:', len(patient_list))
    patient_new_list = check_patient(patient_list)
    print('number of patients after check:', len(patient_new_list))

    #write patient paths into pickle
    # save paths on hard disk with pickle
    with open(save_path, 'wb+') as pathfile:
        pickle.dump(patient_new_list, pathfile)



def load_patient_paths(load_path = '../patient-paths/patients_1_5T.pkl', shuffle = True, seed = None):
    #load all the paths into a list from the pickle file
    with open(load_path, 'rb') as pathfile:
        patient_list = pickle.load(pathfile)

    if shuffle:
        if seed is not None:
            random.seed(seed)
            random.shuffle(patient_list)
            random.seed(None)
        else:
            random.shuffle(patient_list)

    return patient_list

def dataset_split(patient_path_list, k, i, test_num = 100, validation_num = 30):

    test_path = patient_path_list[:test_num]
    train_val_path_list = patient_path_list[test_num:]
    train_path = []
    if k is not None:
        steps = int(len(train_val_path_list) // k)
        data_list = [train_val_path_list[(i*steps):((i+1)*steps)] for i in range(k)]
        validation_path = random.sample(data_list[i], k=validation_num)
        sel = set(range(k)) - {i}
        res = [data_list[j] for j in sel]
        for path in res:
            train_path += path
    else:
        validation_path = train_val_path_list[:validation_num]
        train_path = train_val_path_list[validation_num:]

    print('training num:', len(train_path))
    print('val num:', len(validation_path))
    print('testing num:', len(test_path))
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
    new_shape = bg.shape + (1,)
    bg = bg.reshape(new_shape)
    lt = lt.reshape(new_shape)
    vat = vat.reshape(new_shape)
    scat = scat.reshape(new_shape)
    label_mat = bg
    label_mat = np.concatenate((label_mat, lt), axis = 3)
    #label_mat = lt
    label_mat = np.concatenate((label_mat, vat), axis = 3)
    label_mat = np.concatenate((label_mat, scat), axis = 3)

    #label_mat = process_labels(label_mat)



    return label_mat


def get_batch_from_buffer(batch_size, patient_buffer, cropsize_X, cropsize_Y, border = 20):
    '''
    get a batch of patches, patches are cropped from patients, for each get_batch call only a certain num
    of patients are considered, these patients are provided in patient buffer 
    :param batch_size: 
    :param patient_buffer:  
    :param patient_index: 
    :param stride:  
    :param cropsize: 
    :param batches_per_buffer: 
    :return: 
    '''
    positions = []
    item_idx = 0
    x_shape = (batch_size,) + 3 * (cropsize_X,) + (1,)
    y_shape = (batch_size,) + 3 * (cropsize_Y,) + (4,) #4 is num of classes
    batch_x = np.ndarray(x_shape)
    batch_y = np.ndarray(y_shape)
    crop_per_patient = batch_size // len(patient_buffer)
    '''
    use the pre-written augmentation in libs.preprocessing
    augmentation(dim, patient, label, samples_size, cropsize_X, cropsize_Y, border)
    '''
    if crop_per_patient is not 0:
        for i in range(len(patient_buffer)):
            current_patient = load_data(patient_buffer[i])['img']
            current_patient = current_patient.reshape(current_patient.shape + (1,))
            current_label = merge_labels(patient_buffer[i])
            batch_x[crop_per_patient*i:crop_per_patient*(i+1), ...], pos_X, \
            batch_y[crop_per_patient*i:crop_per_patient*(i+1), ...] = \
                augmentation(3, current_patient, current_label, crop_per_patient, cropsize_X, cropsize_Y, border)
            positions += pos_X
        item_idx = len(patient_buffer)*batch_size
    # if batch_size < patient_buffer capacity, crop_per_patient = 0, just random pick patients
    remainder = batch_size - crop_per_patient*len(patient_buffer)
    for i in range(remainder):
        rnd = np.random.randint(len(patient_buffer))
        current_patient = load_data(patient_buffer[rnd])['img']
        current_patient = current_patient.reshape(current_patient.shape + (1,))
        current_label = merge_labels(patient_buffer[rnd])
        batch_x[item_idx, ...], pos_X, \
        batch_y[item_idx, ...] = augmentation(3, current_patient, current_label, crop_per_patient, cropsize_X, cropsize_Y, border)
        positions += pos_X
        item_idx += 1

    positions = np.array(positions)


    return batch_x, positions, batch_y


def load_patient_buffer(patient_list, patient_index, capacity):
    '''
    load a new batch of patients, if it reaches the end of the patient list, just load the remaining
    :param patient_list: 
    :param patient_index: load patients start from the index
    :param capacity: num of patients to load
    :return: a list containing all the loaded patients
    '''
    if (patient_index + capacity) > len(patient_list):
        patient_buffer = patient_list[patient_index:]
    else:
        patient_buffer = patient_list[patient_index:(patient_index + capacity)]
    return patient_buffer


def get_batch_start_patient(batch_num, batches_per_load, capacity, data_num):
    load_times = batch_num // batches_per_load
    loads_per_epoch = data_num // capacity + 1
    patient_idx = capacity * (load_times % loads_per_epoch)

    return patient_idx


def get_batch(patient_list, capacity, batch_size, cropsize_X, cropsize_Y, batches_per_load, border = 20):
    global batch_num
    patient_idx = get_batch_start_patient(batch_num, batches_per_load, capacity, len(patient_list))
    patient_buffer = load_patient_buffer(patient_list, patient_idx, capacity)
    batch_x, positions, batch_y = get_batch_from_buffer(batch_size, patient_buffer, cropsize_X, cropsize_Y, border = border)
    batch_num += 1

    return batch_x, positions, batch_y


def data_generator(patient_list, capacity, batch_size, cropsize_X, cropsize_Y, batches_per_load, border, mult_inputs):
    while True:
        batch_X, pos, batch_Y = get_batch(patient_list, capacity, batch_size, cropsize_X, cropsize_Y, batches_per_load, border)

        batch_X_dict = {'input_X': batch_X}
        batch_Y_dict = {'output_Y': batch_Y}
        if mult_inputs:
            batch_X_dict['input_position'] = pos

        yield batch_X_dict, batch_Y_dict


def data_clean(rt_input, rt_output):
    for root, dirs, files in os.walk(rt_input):
        for names in filter(lambda name: name[0] != '.', files):

            data_dict = load_data(os.path.join(root, names))
            #print(data_dict['img'].shape)
            data_dict.pop('info', None)         #remove the 'info' field because its length surpassing 31 chars, which cannot be saved using sio.savemat
            label = merge_labels(os.path.join(root, names))
            clean_label = process_label(label)
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
            data_dict['P_BG'] = clean_label[:,:,:,0]
            data_dict['P_LT'] = clean_label[:, :, :, 1]
            data_dict['P_VAT'] = clean_label[:, :, :, 2]
            data_dict['P_AT'] = clean_label[:, :, :, 3]

            a = os.path.split(root)
            pat_id = a[1]
            pat_category = os.path.basename(a[0])  # '1_5T' or '3_T'
            save_path = os.path.join(os.path.join(rt_output, pat_category), pat_id)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            print('write to:', os.path.join(save_path, names))
            sio.savemat(os.path.join(save_path, names), data_dict, do_compression= True)

def get_num_mat(num_mat_path):
    return sio.loadmat(num_mat_path)['n']

def show_slice_num(img, num_mat, idx):
    length = len(str(idx))
    channels = img.shape[2]
    fout = img
    for i in reversed(range(length)):
        digit = idx // (10 ** i)
        idx = idx - digit * (10 ** i)
        numimage = num_mat[:,:,digit]
        numimage = numimage.reshape(numimage.shape + (1,))
        if channels != 1:
            numimage = np.repeat(numimage, channels, axis = 2)
        offset = (length - 1 -i) * 6
        fout[0:7, offset : offset + 5, :] = fout[0:7, offset : offset + 5, :] + (numimage / 255.0)
    fout[fout > 1.0] = 1.0
    return fout
