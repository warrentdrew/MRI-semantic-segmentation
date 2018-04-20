'''
Setting for importing the path
'''
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from libs.util import Patient
from utils import load_data, merge_labels
import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib import gridspec

def retrieve_pixel_classes(cls_vector):
    #print('cls vector shape:', cls_vector.shape)
    res = np.where(cls_vector == 1)[0]
    return res


def colorize_for_label_check(prediction, colors={0 : np.array([0,0,0]),     #class 0: background    -> black
                                                 1 : np.array([1,0,0.2]),   #class 1: lean tissue   -> red
                                                 2 : np.array([0,1,0.2]),   #class 2: VAT           -> green
                                                 3 : np.array([0.1,0.1,1]),  #class 3: SCAT          -> blue
                                                 4 : np.array([1,1,1]),      #no class               -> white
                                                 5 : np.array([1,1,0])}):   #two or more classes            -> yellow
    """Colorize for patient-plots."""
    pred_picture = np.zeros(shape= prediction.shape[:3] + (3,))
    for x , row in enumerate(prediction):
        for y, col in enumerate(row):
            for z, prd in enumerate(col):
                res = retrieve_pixel_classes(prd)
                if len(res) is 1:                   #len(res) shows how many classes the pixel belongs to
                    pred_picture[x,y,z,:] = colors[res[0]]      #res[0] shows which class the pixel belongs to
                elif len(res) is 0:
                    pred_picture[x, y, z, :] = colors[4]
                elif len(res) is 2:
                    pred_picture[x, y, z, :] = colors[5]
                else:
                    print('more than 2 classes')
                    pred_picture[x, y, z, :] = colors[5]

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

    def check_slices_and_labels(self, dim, alpha):              #this dim is more like a position information
        '''
        Used only for checking the label mask and image data
        :param dim: 
        :param alpha: 
        :return: 
        '''
        # get patient data
        img, label = self.get_slices(count=False)
        #niis = np.argmax(label, axis=-1)  # originally the label was a probability in each mask for every position
                                           # returns the index for the max value in the last axis for the niis, which is
                                            # the class each pixel belongs to can be [0, 1, 2]
        '''
        Don't need to look for the argmax in masks, we want to now plot out all the masks which has '1'
        in the position
        '''
        print('img shape:', img.shape)
        print('label shape:', label.shape)


        x, y, z = img.shape[:3]
        # arrays for plotting
        img = (img).astype('float32')
        label = (label).astype('float32')

        # channel: only 1 channel


        # plot cuts in each dim
        fig = plt.figure(figsize=(18, 7))
        #gs = gridspec.GridSpec(1, 3,
        #                       width_ratios=[x / y, z / y, x / z],
        #                       height_ratios=[1])
        #plt.subplot(211)
        #plt.axis('off')
        # plt.xlabel('x-axis')
        # plt.ylabel('y-axis')
        print('test:', dim[2])
        img1 = img[:, :, dim[2]].reshape(img.shape[:2])
        print('img shape:', img1.shape)
        plt.imshow(img1, cmap='gray')
        #check1 = colorize_for_label_check(label)[:, :, dim[2], :]
        #print('check1 shape', check1.shape)
        #plt.show()
        #ax1.imshow(np.fliplr(np.rot90(colorize_for_label_check(label)[:, :, dim[2], :], axes=(1, 0))), interpolation='none', alpha=alpha)
        #ax2 = plt.subplot(gs[1])
        #plt.axis('off')

        #ax2.imshow(img[dim[0], :, :, chNr], interpolation='none', cmap='gray')
        #ax2.imshow(colorize_for_label_check(label)[dim[0], :, :, :], interpolation='none', alpha=alpha)
        #ax3 = plt.subplot(gs[2])
        #plt.axis('off')
        #ax3.imshow(np.fliplr(np.rot90(img[:, dim[1], :, chNr], axes=(1, 0))), interpolation='none', cmap='gray')
        #ax3.imshow(np.fliplr(np.rot90(colorize_for_label_check(label)[:, dim[1], :, :], axes=(1, 0))), interpolation='none', alpha=alpha)

        return plt




def load_correct_patient(train_path, validation_path, test_path, forget_slices):
    '''
    preparation for the fit function
    1. creating the Patient instances and store then in the ring buffer
    2. creating the Patients matrix for the validation data
    :param train_path: 
    :param validation_path: 
    :param test_path: 
    :return: 
    '''
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
        patients_val.append(patient_test)

    # get slices for validation data
    patients_val_slices = []
    for patient in patients_val:
        slices = patient.get_slices(verbose=False)  #slice is a tuple of (data, label)
        patients_val_slices.append(slices)

    return (patients_train, patients_val, patients_test, patients_val_slices)
