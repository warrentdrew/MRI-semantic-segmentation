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
#import matplotlib
#matplotlib.use('TkAgg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
from matplotlib import gridspec
import math
import cv2
from skimage import color



def colorize(prediction, colors={0 : np.array([0,0,0]),     #class 0: background    -> black[0,0,0]
                                 1 : np.array([0.2,1,0]),   #class 1: lean tissue   -> red[1,0,0.2]        green
                                 2 : np.array([1,1,0.2]),   #class 2: VAT           -> green[0,1,0.2]      yellow
                                 3 : np.array([1,0.2,0])  #class 3: SCAT          -> blue[0.1,0.1,1]     red
                                }):
    """Colorize for patient-plots."""
    #prediction here has dim 3
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

        return plt, fig

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


    def plot_prediction_vs_ground_truth(self, depth, model, model_name):
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
        head1, tail1 = os.path.split(self.patient_path)
        head2, tail2 = os.path.split(head1)
        _, tail3 = os.path.split(head2)
        id = os.path.join(os.path.join(tail3, tail2), tail1)
        return id

    # def plot_mosaic_style(self, col): # this plots 3 images in mosaic style(for each slice): originals, predictions, labels
    #     img, label = self.get_slices(count=False)
    #     img_shape = img.shape[:2]
    #
    #     #print('test', img.shape)
    #     #label = np.argmax(label, axis=-1)  # shape is x,y,z,1
    #
    #     #prediction = self.predict_patient(model)
    #     #prediction = np.argmax(prediction, axis=-1)
    #     #prediction = colorize(prediction)
    #     slice_num = img.shape[2]
    #     print("slice number:", slice_num)
    #     """
    #     res_dict_ls = []
    #     for i in range(slice_num):
    #         dict = {}
    #         dict["img"] = img[:,:,i].reshape(img_shape)
    #         dict["label"] = colorize(label[:,:,i])
    #         dict["prediction"] = colorize(prediction[:,:,i])
    #         res_dict_ls.append(dict)
    #     """
    #
    #     fig = plt.figure()
    #     row = math.ceil(slice_num / col)
    #     gs = gridspec.GridSpec(row, col)
    #     slice_idx = 0
    #     for i in range(row):
    #         for j in range(col):
    #             if slice_idx < slice_num:
    #                 ax = plt.subplot(gs[i, j])
    #                 ax.axis('off')
    #                 ax.imshow(img[:,:,slice_idx].reshape(img_shape))
    #             else:
    #                 ax = plt.subplot(gs[i, j])
    #                 ax.axis('off')
    #                 ax.imshow(np.zeros(img_shape))
    #             slice_idx += 1
    #
    #     return plt

    def save_mosaic_style(self, col, space = 0, back = 0):
        img, label = self.get_slices(count=False) #img shape x,y,z,1
        (h, w, slices, _) = img.shape

        #compute num of rows and cols and spaces
        row = math.ceil(slices / col)
        # if space is not 0:
        #     vspace = back * (np.ones([h, space]))       #vspace stands for the vertical space on columns
        #     hspace = back * (np.ones([space, (col * w + (col -1 ) * space)]))
        # else:
        #     vspace = hspace = 0

        slice_idx = 0
        for i in range(row):
            for j in range(col):
                if slice_idx < slices:
                    f = img[:,:,slice_idx].reshape(img.shape[:2])
                    data = f
                    #f = show_numbers(f, slice_idx)
                else:
                    data = np.zeros([h,w])

                if j == 0:
                    R = data
                else:
                    if space is not 0:
                        vspace = back * (np.ones([h, space]))  # vspace stands for the vertical space on columns
                        R = np.concatenate([R,vspace,data], axis=1)
                    else:
                        R = np.concatenate([R,data], axis=1)
                slice_idx += 1

            if i == 0:
                g = R
            else:
                if space is not 0:
                    hspace = back * (np.ones([space, (col * w + (col - 1) * space)]))
                    g = np.concatenate([g, hspace, R], axis=0)
                else:
                    g = np.concatenate([g, R], axis=0)


        fig = plt.figure(figsize=(100, 100), dpi=100)
        plt.axis("off")
        plt.imshow(g, cmap="gray")
        fig.savefig("../img-test/newres.png")
        plt.show()














    def predict_patient(self, model):
        """Prediction of patient with special model."""
        dicoms, _ = self.get_slices(count=False)
        dishape = dicoms.shape
        print('dishape:', dishape)
        outshape = model.get_output_shape_at(-1)[1:]
        # inshape = model.internal_input_shapes
        inshape = model.get_input_shape_at(0)
        if len(inshape) == 1:
            feedpos = False
        elif len(inshape) > 1:
            feedpos = True
        else:
            raise 'no input'

        # inshape = inshape[0][1:]
        inshape = inshape[1:]  # input shape (32,32,32,1) output shape(32,32,32,4) dishape(256,192,105,1)
        print('inshape:', inshape)
        prediction_shape = dicoms.shape[:3] + (outshape[-1],)
        prediction = np.zeros(prediction_shape)
        prediction[:, :, :, 0] = np.ones(dicoms.shape[
                                         :3])  # the default is that all the pixels are predicted as background, so the remainer during the cropping are classified as bg
        deltas = tuple((np.array(inshape[:3]) - np.array(outshape[
                                                         :3])) // 2)  # this is like the padding added outside the output of the input, to make it have the same shape as the input
        input_dict = {}
        #add a margin to make the prediction in the middle
        margin_x = (dishape[0] - dishape[0] // inshape[0] * inshape[0] ) // 2
        margin_y = (dishape[1] - dishape[1] // inshape[1] * inshape[1] ) // 2
        margin_z = (dishape[2] - dishape[2] // inshape[2] * inshape[2]) // 2
        print('margin x: {}, y:{}, z:{}'.format(margin_x,margin_y,margin_z))
        for x in range(dishape[0] // inshape[0]):
            for y in range(dishape[1] // inshape[1]):
                for z in range(dishape[2] // inshape[2]):

                    input_dict['input_X'] = np.expand_dims(dicoms[x*inshape[0] + margin_x : (x +1) * inshape[0] + margin_x,
                                                           y * inshape[1] + margin_y: (y + 1) * inshape[1] + margin_y,
                                                           z * inshape[2] + margin_z: (z + 1) * inshape[2] + margin_z],
                                                           axis=0)
                    if feedpos:
                        (size_x, size_y, size_z) = dishape[:3]
                        cropsize_X = inshape[0]
                        # hardcoded, see train-script
                        border = 20
                        max_pos = np.array(
                            [size_x - cropsize_X - border, size_y - cropsize_X - border, size_z - cropsize_X - border])
                        pos = np.array([x, y, z]) / max_pos
                        input_dict['input_position'] = np.expand_dims(pos, axis=0)

                    prediction[x*inshape[0] + margin_x + deltas[0]: x*inshape[0] + margin_x + deltas[0] + outshape[0],
                    y * inshape[1] + margin_y + deltas[1]: y * inshape[1] + margin_y + deltas[1] + outshape[1],
                    z * inshape[2] + margin_z + deltas[2]: z * inshape[2] + margin_z + deltas[2] + outshape[2], :] = model.predict(
                        input_dict)  # this ensures that only the most outside part the for whole image is predicted as bg
        return prediction




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
        patients_test.append(patient_test)

    # get slices for validation data
    patients_val_slices = []
    for patient in patients_val:
        slices = patient.get_slices(verbose=False)  #slice is a tuple of (data, label)
        patients_val_slices.append(slices)

    return (patients_train, patients_val, patients_test, patients_val_slices)
