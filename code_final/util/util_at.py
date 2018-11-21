'''
Setting for importing the path
'''
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from util.util import Patient
import matplotlib.pyplot as plt
import math
import time
import scipy.io as sio
import numpy as np
import os
import pickle
import random
from skimage.morphology import closing, cube

'''
label processing: subtraction and morphological closing
'''
def retrieve_pixel_classes(cls_vector):
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


def class_map_zero_pixel_closing(class_map, zero_pixel_pos_list,range):

    increase_range = True
    if len(zero_pixel_pos_list) == 0:
        return class_map
    else:
        clean_map = class_map
        class_map = closing(class_map, cube(range)).astype(np.int8)
        for (x,y,z) in zero_pixel_pos_list:
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
                label_map[x,y,z,:] = val_to_one_hot_array((class_map[x,y,z]-1), 4)

    return label_map

def process_label(label_mat):
    double_class_removed_map, zero_pixel_pos_list = label_remove_2_classes(label_mat)
    class_map = class_map_zero_pixel_closing(double_class_removed_map, zero_pixel_pos_list, 2).astype(np.int8)
    clean_label = retrieve_label_from_class_map(class_map)
    return clean_label

def data_clean(rt_input, rt_output):
    for root, dirs, files in os.walk(rt_input):
        for names in filter(lambda name: name[0] != '.', files):

            data_dict = load_data(os.path.join(root, names))
            #print(data_dict['img'].shape)
            data_dict.pop('info', None)         #remove the 'info' field because its length surpassing 31 chars, which cannot be saved using sio.savemat
            label = merge_labels(os.path.join(root, names))
            clean_label = process_label(label)

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

'''
store and load patients, load 3T patients for validation
'''
def store_patients(data_path, save_path = '../patient-paths/patients_1_5T.pkl'):
    #load patient paths from dataset
    patient_list = []
    for root, dirs, files in os.walk(data_path):
        for names in filter(lambda name: name[0] != '.', files):
            print('name test:', os.path.join(root, names))
            patient_list.append(os.path.join(root, names))
    print('number of all patients:', len(patient_list))
    #write patient paths into pickle
    # save paths on hard disk with pickle
    with open(save_path, 'wb+') as pathfile:
        pickle.dump(patient_list, pathfile)

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

def dataset_split(patient_path_list, k, i, last_val_patients, test_num = 100, validation_num = 30, train_num = 150, seed = None):

    #drop last validation patients
    if last_val_patients is not None:
        for patient in last_val_patients:
            patient.drop()

    test_path = patient_path_list[:test_num]
    train_val_path_list = patient_path_list[test_num:]
    train_path = []

    if seed is not None:
        random.seed(seed)

    if k is not None:
        steps = int(len(train_val_path_list) // k)
        data_list = [train_val_path_list[(i*steps):((i+1)*steps)] for i in range(k)]
        validation_path = random.sample(data_list[i], k=validation_num)
        sel = set(range(k)) - {i}
        res = [data_list[j] for j in sel]
        for path in res:
            train_path += path
        print("numtest", train_num)
        train_path = random.sample(train_path, k=train_num)
    else:
        validation_path = train_val_path_list[:validation_num]
        train_path = train_val_path_list[validation_num:]
        train_path = random.sample(train_path, k=train_num)

    print('training num:', len(train_path))
    print('val num:', len(validation_path))
    print('testing num:', len(test_path))
    return train_path, validation_path, test_path

def load_data(path):
    return sio.loadmat(path)

def merge_labels(path):
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
    label_mat = np.concatenate((label_mat, vat), axis = 3)
    label_mat = np.concatenate((label_mat, scat), axis = 3)
    return label_mat

def get_3t_val_paths(num, seed = None, path_3t = '/home/d1251/no_backup/d1251/patient_paths/patients_3T.pkl'):
    val_list = load_patient_paths(load_path=path_3t, shuffle= False, seed=None)
    if seed is not None:
        random.seed(seed)
    if num is not None:
        val_list = random.sample(val_list, num)
    return val_list

'''
save mosaic style plot
'''
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

def get_patient_id(patient_path):
    head1, tail1 = os.path.split(patient_path)
    head2, tail2 = os.path.split(head1)
    _, tail3 = os.path.split(head2)
    id = os.path.join(os.path.join(tail3, tail2),tail1)
    return id

'''
New Positional Encoding
'''
def pos_slice_expand(pos, slice_num):
    ret = np.zeros([slice_num, 4, 4, 3])
    pos_row = []
    pos_col = []
    pos_slice = []
    result = []
    tmp = np.zeros((3,))

    for item in pos:
        for i in range(0, 32, 8): #expand row dim    shape is now in a slice first version [slice, h, w, channel]
            tmp[...] = item[...]
            tmp[1] = tmp[1] + i
            k = np.zeros((1, 3))
            k[...] = tmp.reshape((1,) + tmp.shape)
            pos_row.append(k)
        ret[0,0,:,:] = np.concatenate(pos_row, axis = 0) #result shape [1, 1, 4, 3]
        pos_row = []
        for i in range(ret.shape[2]):      #hard coded
            for j in range(0, 32, 8): #expand row dim    shape is now in a slice first version [slice, h, w, channel]
                tmp[...] = ret[0,0,i,:]
                tmp[0] = item[0] + j
                k = np.zeros((1, 3))
                k[...] = tmp.reshape((1,) + tmp.shape)
                pos_col.append(k)

            ret[0,:,i,:] = np.concatenate(pos_col, axis = 0) # shape [1,4,1,3] ret shape [1, 4, 4, 3]
            pos_col = []

        for i in range(slice_num):
            k = np.zeros((1, 4, 4, 3))
            k[...] = ret[0, ...]
            k[...,-1] = k[...,-1] + i
            pos_slice.append(k)
        ret = np.concatenate(pos_slice, axis = 0) #ret shape [32, 4, 4, 3]
        pos_slice = []
        result.append(ret)
    return result

'''
Patient_AT and auxiliary functions
'''

def colorize(prediction, colors={0 : np.array([0,0,0]),     #class 0: background    -> black[0,0,0]
                                 1 : np.array([0.2,1,0]),   #class 1: lean tissue   -> red[1,0,0.2]        green
                                 2 : np.array([1,1,0.2]),   #class 2: VAT           -> green[0,1,0.2]      yellow
                                 3 : np.array([1,0.2,0])  #class 3: SCAT          -> blue[0.1,0.1,1]     red
                                }):
    """Colorize for patient-plots."""
    #prediction here need to have dim 3
    pred_picture = np.zeros(shape= prediction.shape[:2] + (3,))
    for x , row in enumerate(prediction):
        for y, col in enumerate(row):
                pred_picture[x, y, :] = colors[int(prediction[x,y,...])]

    return pred_picture

'''
patient class for AT are inherited from the original patient class 
'''
class Patient_AT(Patient):
    def __init__(self, patient_path, forget_slices):
        Patient.__init__(self, dicom_dirs=None, nii_dirs=None, forget_slices=forget_slices)
        self.patient_path = patient_path

    def load_slices(self, verbose=False):
        """Load patients data  in numpy arrays."""

        train = load_data(self.patient_path)['img']
        train_shape = train.shape + (1,)
        train = train.reshape(train_shape)  # change the shape into (192, 256, 105, 1)
        label = merge_labels(self.patient_path)


        return (train, label)

    def check_slice(self, depth):              #this dim is more like a position information

        # get patient data
        img, label = self.get_slices(count=False)

        print('img shape:', img.shape)
        print('label shape:', label.shape)


        x, y, z = img.shape[:3]
        # arrays for plotting
        img = (img).astype('float32')
        label = (label).astype('float32')

        # channel: only 1 channel

        fig = plt.figure()

        img1 = img[:, :, depth].reshape(img.shape[:2])
        plt.axis("off")
        plt.imshow(img1, cmap='gray')

        return plt, fig


    def plot_prediction_vs_ground_truth(self, depth, model, model_name, border):
        """Plot prediction (left) vs ground truth (right)."""
        img, label  = self.get_slices(count=False)

        img = img[:,:,depth].reshape(img.shape[:2])
        print('test', img.shape)
        labeled_slice = np.argmax(label[:,:,depth,:], axis=-1) #shape is x,y,1,1
        labeled_slice = colorize(labeled_slice)

        #prediction = np.argmax(predict_patient(patient,model)[:,:,depth,:], axis=-1)
        prediction = self.predict_patient(model, border)

        prediction =  np.argmax(prediction[:,:,depth,:], axis=-1)
        prediction = colorize(prediction)

        head1, tail1 = os.path.split(self.patient_path)
        head2, tail2 = os.path.split(head1)
        _, tail3 =  os.path.split(head2)
        patient_name = os.path.join(os.path.join(tail3,tail2),tail1)

        fig = plt.figure(dpi=100)

        fig.suptitle("Model : {}    Patient : {}    Pos : [:,:,{}]".format(model_name, patient_name, depth) , fontsize=14)
        ax1 = fig.add_subplot(1,2,1)
        ax1.set_title('Prediction')
        plt.axis('off')
        ax1.imshow(img, interpolation='none', cmap='gray')
        ax1.imshow(prediction, interpolation='none', cmap='gray', alpha = 0.3)
        ax2 = fig.add_subplot(1,2,2)
        ax2.set_title('Label')
        plt.axis('off')
        ax2.imshow(img, interpolation='none', cmap='gray')
        ax2.imshow(labeled_slice, interpolation='none', cmap='gray', alpha = 0.3)
        return plt

    def plot_img_pred_gt(self, depth, model, model_name):
        """Plot prediction (left) vs ground truth (right)."""
        img, label  = self.get_slices(count=False)

        img = img[:,:,depth].reshape(img.shape[:2])
        print('test', img.shape)
        labeled_slice = np.argmax(label[:,:,depth,:], axis=-1) #shape is x,y,1,1
        labeled_slice = colorize(labeled_slice)

        #prediction = np.argmax(predict_patient(patient,model)[:,:,depth,:], axis=-1)
        prediction = self.predict_patient(model)

        prediction =  np.argmax(prediction[:,:,depth,:], axis=-1)
        prediction = colorize(prediction)

        head1, tail1 = os.path.split(self.patient_path)
        head2, tail2 = os.path.split(head1)
        _, tail3 =  os.path.split(head2)
        patient_name = os.path.join(os.path.join(tail3,tail2),tail1)

        fig = plt.figure(dpi=100)

        fig.suptitle("Model : {}    Patient : {}    Pos : [:,:,{}]".format(model_name, patient_name, depth) , fontsize=14)
        ax1 = fig.add_subplot(1,3,1)
        ax1.set_title('Image')
        plt.axis('off')
        ax1.imshow(img, interpolation='none', cmap='gray')

        ax2 = fig.add_subplot(1,3,2)
        ax2.set_title('Prediction')
        plt.axis('off')
        ax2.imshow(img, interpolation='none', cmap='gray')
        ax2.imshow(prediction, interpolation='none', cmap='gray', alpha = 0.3)

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.set_title('Label')
        plt.axis('off')
        ax3.imshow(img, interpolation='none', cmap='gray')
        ax3.imshow(labeled_slice, interpolation='none', cmap='gray', alpha=0.3)

        return plt

    def get_patient_id(self):
        head1, _ = os.path.split(self.patient_path)
        head2, tail2 = os.path.split(head1)
        _, tail3 = os.path.split(head2)
        id = os.path.join(tail3, tail2)
        return id

    def predict_patient(self, model, border, add_pos = False, add_prh = False):
        """Prediction of patient with special model."""
        #original implementation neglect the remaining slices (or w, h) at the end
        #new implementation add zero paddings at the boundary and remove them after prediction
        #this func is now only suitable for input shape = output shape
        #calculate the time for making prediction

        imgs, _ = self.get_slices(count=False)
        imgshape = imgs.shape
        start = time.time()

        if add_pos:
            inshape = model.get_input_shape_at(0)[0]  # if pos is added, add [0] at the end
        else:
            inshape = model.get_input_shape_at(0)

        if add_prh:
            outshape = model.get_output_shape_at(-1)[1][1:]  # add [1] in middle for multi output
        else:
            outshape = model.get_output_shape_at(-1)[1:]

        if len(inshape) == 1:
            feedpos = False
        elif len(inshape) > 1:
            feedpos = True
        else:
            raise 'no input'

        inshape = inshape[1:]  # input shape (32,32,32,1) output shape(32,32,32,4) imgshape(256,192,105,1)
        #print('inshape:', inshape)
        deltas = tuple((np.array(inshape[:3]) - np.array(outshape[:3])) // 2)
        # this is like the padding added outside the output of the input, to make it have the same shape as the input
        input_dict = {}
        #add a margin to make the prediction in the middle
        margin_x = math.ceil(imgshape[0] / inshape[0]) * inshape[0] - imgshape[0]
        margin_x_front = margin_x // 2
        margin_x_end = margin_x - margin_x_front
        margin_y = math.ceil(imgshape[1] / inshape[1]) * inshape[1] - imgshape[1]
        margin_y_front = margin_y // 2
        margin_y_end = margin_y - margin_y_front
        margin_z = math.ceil(imgshape[2] / inshape[2]) * inshape[2] - imgshape[2]
        margin_z_front = margin_z // 2
        margin_z_end = margin_z - margin_z_front

        imgs = np.pad(imgs.reshape(imgs.shape[:3]), ((margin_x_front, margin_x_end), (margin_y_front, margin_y_end), (margin_z_front, margin_z_end)),
                      mode='constant', constant_values= 0)
        imgs = imgs.reshape(imgs.shape + (1,))

        (shape_x, shape_y, shape_z, _) = newshape = imgs.shape

        prediction_shape = imgs.shape[:3] + (outshape[-1],)
        print('prediction shape:', prediction_shape)
        prediction = np.zeros(prediction_shape)
        prediction[:, :, :, 0] = np.ones(imgs.shape[:3])
        # the default is that all the pixels are predicted as background, so the remainer during the cropping are classified as bg
        for x in range(0, shape_x, inshape[0]):
            for y in range(0, shape_y, inshape[1]):
                for z in range(0, shape_z, inshape[2]):

                    input_dict['input_X'] = np.expand_dims(imgs[x : x +inshape[0],
                                                           y : y + inshape[1],
                                                           z : z + inshape[2],:],
                                                           axis=0)

                    if feedpos:
                        (size_x, size_y, size_z) = imgshape[:3]
                        cropsize_X = inshape[0]
                        # hardcoded, see train-script
                        #border = 20
                        max_pos = np.array([size_x - cropsize_X - border, size_y - cropsize_X - border, size_z - cropsize_X - border])
                        pos = np.array([x, y, z]) / max_pos
                        input_dict['input_position'] = np.expand_dims(pos, axis=0)

                    prediction[x + deltas[0] : x + deltas[0] + outshape[0],
                               y + deltas[1] : y + deltas[1] + outshape[1],
                               z + deltas[2] : z + deltas[2] + outshape[2],:] = model.predict(input_dict)

        prediction = prediction[margin_x_front : shape_x-margin_x_end, margin_y_front : shape_y-margin_y_end, margin_z_front : shape_z-margin_z_end]
        end = time.time()
        duration = end - start
        print('time for pred is:', duration)

        return prediction


def load_correct_patient(train_path, validation_path, test_path, forget_slices):

    patients_train = []
    patients_val = []
    patients_test = []

    for pat_train in train_path:
        patient_train = Patient_AT(patient_path= pat_train, forget_slices = forget_slices)
        patients_train.append(patient_train)
    for pat_val in validation_path:
        patient_val = Patient_AT(patient_path= pat_val, forget_slices = forget_slices)
        patients_val.append(patient_val)
    for pat_test in test_path:
        patient_test = Patient_AT(patient_path= pat_test, forget_slices = forget_slices)
        patients_test.append(patient_test)

    # get slices for validation data
    patients_val_slices = []
    for patient in patients_val:
        slices = patient.get_slices(verbose=False)  #slice is a tuple of (data, label)
        patients_val_slices.append(slices)

    return (patients_train, patients_val, patients_test, patients_val_slices)
