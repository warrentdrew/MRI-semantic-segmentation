'''
Setting for importing the path
'''
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from libs.util import Patient
from utils import load_data, merge_labels, show_slice_num, get_num_mat
import numpy as np
#import matplotlib
#matplotlib.use('TkAgg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import math
from skimage import color
import time




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

    def save_mosaic_style(self, model, model_name, col, border, space = 0, back = 0, save_rt = '/home/d1251/Downloads/'):
        img, label = self.get_slices(count=False) #img shape x,y,z,1
        (h, w, slices, _) = img.shape

        label = np.argmax(label, axis=-1)  # shape is x,y,1,1
        #labeled_slice = colorize(labeled_slice)

        prediction = self.predict_patient(model, border)
        prediction = np.argmax(prediction, axis=-1)
        #prediction = colorize(prediction)

        #compute num of rows and cols and spaces
        row = math.ceil(slices / col)
        slice_idx = 0
        num_mat = get_num_mat('/home/d1251/no_backup/d1251/aux_file/numbers.mat')

        for i in range(row):
            for j in range(col):
                if slice_idx < slices:
                    f_img = img[:,:,slice_idx].reshape([h,w,1])
                    f_label = colorize(label[:,:,slice_idx])
                    f_pred = colorize(prediction[:,:,slice_idx])
                    f_img = show_slice_num(f_img, num_mat, slice_idx+1).reshape([h,w])
                    f_label = show_slice_num(f_label, num_mat, slice_idx + 1)
                    f_pred = show_slice_num(f_pred, num_mat, slice_idx + 1)
                else:
                    f_img = np.zeros([h,w])
                    f_label = f_pred = np.zeros([h,w,3])
                if j == 0:
                    R_img = f_img
                    R_label = f_label
                    R_pred = f_pred
                else:
                    if space is not 0:
                        vspace = back * (np.ones([h, space]))  # vspace stands for the vertical space on columns
                        vspace_3c = back * (np.ones([h, space, 3]))  # vspace stands for the vertical space on columns
                        R_img = np.concatenate([R_img,vspace,f_img], axis=1)
                        R_label = np.concatenate([R_label, vspace_3c, f_label], axis=1)
                        R_pred = np.concatenate([R_pred, vspace_3c, f_pred], axis=1)

                    else:
                        R_img = np.concatenate([R_img,f_img], axis=1)
                        R_label = np.concatenate([R_label, f_label], axis=1)
                        R_pred = np.concatenate([R_pred, f_pred], axis=1)
                slice_idx += 1

            if i == 0:
                g_img = R_img
                g_label = R_label
                g_pred = R_pred
            else:
                if space is not 0:
                    hspace = back * (np.ones([space, (col * w + (col - 1) * space)]))
                    hspace_3c = back * (np.ones([space, (col * w + (col - 1) * space), 3]))
                    g_img = np.concatenate([g_img, hspace, R_img], axis=0)
                    g_label = np.concatenate([g_label, hspace_3c, R_label], axis=0)
                    g_pred = np.concatenate([g_pred, hspace_3c, R_pred], axis=0)
                else:
                    g_img = np.concatenate([g_img, R_img], axis=0)
                    g_label = np.concatenate([g_label, R_label], axis=0)
                    g_pred = np.concatenate([g_pred, R_pred], axis=0)


        patient_id = self.get_patient_id().replace('/', '_')
        print('patient id:', patient_id)
        fig1 = plt.figure(figsize=(50, 100), dpi=100)
        plt.axis('off')
        plt.imshow(g_img, cmap="gray")
        fig1.savefig(os.path.join(save_rt, "{}-img__{}.png".format(model_name, patient_id)))
        plt.close()


        fig2 = plt.figure(figsize=(50, 100), dpi=100)
        plt.axis('off')
        plt.imshow(g_img, cmap="gray")
        plt.imshow(g_pred, interpolation='none', alpha=0.3)
        fig2.savefig(os.path.join(save_rt, "{}-pred__{}.png".format(model_name, patient_id)))
        plt.close()

        fig3 = plt.figure(figsize=(50, 100), dpi=100)
        plt.axis('off')
        plt.imshow(g_img, cmap="gray")
        plt.imshow(g_label, interpolation='none', alpha=0.3)
        fig3.savefig(os.path.join(save_rt, "{}-label__{}.png".format(model_name, patient_id)))
        plt.close()





    def predict_patient(self, model, border):
        """Prediction of patient with special model."""
        #original implementation neglect the remaining slices (or w, h) at the end
        #new implementation add zero paddings at the boundary and remove them after prediction
        #this func is now only suitable for input shape = output shape
        #calculate the time for making prediction

        imgs, _ = self.get_slices(count=False)
        imgshape = imgs.shape
        #print('imgshape:', imgshape)
        start = time.time()
        outshape = model.get_output_shape_at(-1)[1:]
        inshape = model.get_input_shape_at(0)[0]            #if pos is added, add [0] at the end
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
        #print('margin x: {}, y:{}, z:{}'.format(margin_x,margin_y,margin_z))
        margin_z_front = margin_z // 2
        margin_z_end = margin_z - margin_z_front

        imgs = np.pad(imgs.reshape(imgs.shape[:3]), ((margin_x_front, margin_x_end), (margin_y_front, margin_y_end), (margin_z_front, margin_z_end)),
                      mode='constant', constant_values= 0)
        imgs = imgs.reshape(imgs.shape + (1,))

        (shape_x, shape_y, shape_z, _) = newshape = imgs.shape
        #print('new shape:', newshape)

        prediction_shape = imgs.shape[:3] + (outshape[-1],)
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

                    #print('x{},y{},z{}'.format(x,y,z))
                    prediction[x + deltas[0] : x + deltas[0] + outshape[0],
                               y + deltas[1] : y + deltas[1] + outshape[1],
                               z + deltas[2] : z + deltas[2] + outshape[2],:] = model.predict(
                        input_dict)

        #print('origin prediction shape', prediction.shape)
        prediction = prediction[margin_x_front : shape_x-margin_x_end, margin_y_front : shape_y-margin_y_end, margin_z_front : shape_z-margin_z_end]
        #print('final prediction shape', prediction.shape)
        end = time.time()
        duration = end - start
        print('time for pred is:', duration)

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
