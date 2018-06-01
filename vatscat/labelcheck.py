'''

when import patient, matplot lib will not show image
'''
import matplotlib.pyplot as plt
from  utils import  load_data, merge_labels
import numpy as np
from matplotlib import gridspec
import os
import random
from utils import process_label

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

def revised_colorize_for_label_check(prediction, colors={0 : np.array([0,0,0]),     #class 0: background    -> black
                                                 1 : np.array([1,0,0.2]),   #class 1: lean tissue   -> red
                                                 2 : np.array([0,1,0.2]),   #class 2: VAT           -> green
                                                 3 : np.array([0.1,0.1,1]),  #class 3: SCAT          -> blue
                                                 4 : np.array([1,1,1]),      #no class               -> white
                                                 5 : np.array([1,1,0])}):   #two or more classes
    """Colorize for patient-plots."""
    pred_picture = np.zeros(shape=prediction.shape[:3] + (3,))
    for x, row in enumerate(prediction):
        for y, col in enumerate(row):
            for z, prd in enumerate(col):
                res = retrieve_pixel_classes(prd)
                if len(res) is 1:  # len(res) shows how many classes the pixel belongs to
                    pred_picture[x, y, z, :] = colors[res[0]]  # res[0] shows which class the pixel belongs to
                elif len(res) is 0:
                    pred_picture[x, y, z, :] = colors[4]
                elif len(res) is 2:
                    newres = np.delete(res, np.where(res == 3))
                    if len(newres) is 1:
                        pred_picture[x, y, z, :] = colors[newres[0]]
                    else:
                        pred_picture[x, y, z, :] = colors[5]
                else:
                    print('more than 2 classes')
                    pred_picture[x, y, z, :] = colors[5]

    return pred_picture



def check_slices_and_labels(path, pos, alpha):  # this dim is more like a position information

    # get patient data
    img = load_data(path)['img']
    label = merge_labels(path)

    clean_label = process_label(label)


    print('img shape:', img.shape)
    print('label shape:', label.shape)

    x, y, z = img.shape[:3]
    # arrays for plotting
    img = (img).astype('float32')
    label = (label).astype('float32')

    # channel: only 1 channel


    # plot cuts in each dim
    fig = plt.figure(figsize=(18, 7))
    gs = gridspec.GridSpec(1, 3,
                           width_ratios=[x / y, z / y, x / z],
                           height_ratios=[1])
    ax1 = plt.subplot(231)
    plt.axis('off')
    #ax1.imshow(img[:, :, pos[2]], interpolation='none', cmap='gray')
    ax1.imshow(colorize_for_label_check(img)[:, :, pos[2], :], interpolation='none', alpha = alpha)

    ax2 = plt.subplot(232)
    plt.axis('off')
    #ax2.imshow(np.rot90(img[pos[0], :, :]), interpolation='none', cmap='gray')
    ax2.imshow(np.rot90(colorize_for_label_check(img)[pos[0], :, :, :]), interpolation='none', alpha = alpha)

    ax3 = plt.subplot(233)
    plt.axis('off')
    #ax3.imshow(np.rot90(img[:, pos[1], :]), interpolation='none', cmap='gray')
    ax3.imshow(np.rot90(colorize_for_label_check(img)[:, pos[1], :, :]), interpolation='none', alpha = alpha)

    ax4 = plt.subplot(234)
    plt.axis('off')
    #ax4.imshow(img[:, :, pos[2]], interpolation='none', cmap='gray')
    ax4.imshow(colorize_for_label_check(label)[:, :, pos[2], :], interpolation='none', alpha = alpha)

    ax5 = plt.subplot(235)
    plt.axis('off')
    #ax5.imshow(np.rot90(img[pos[0], :, :]), interpolation='none', cmap='gray')
    ax5.imshow(np.rot90(colorize_for_label_check(label)[pos[0], :, :, :]), interpolation='none', alpha = alpha)

    ax6 = plt.subplot(236)
    plt.axis('off')
    #ax6.imshow(np.rot90(img[:, pos[1], :]), interpolation='none', cmap='gray')
    ax6.imshow(np.rot90(colorize_for_label_check(label)[:, pos[1], :, :]), interpolation='none', alpha = alpha)





    return plt


patient_path = '/med_data/Segmentation/AT/1_5T/PLIS2_3529_TL/rework.mat'
#patient = Patient_AT(patient_path=patient_path, forget_slices=False)
'''

data_path = '/med_data/Segmentation/AT/1_5T/'
patient_list = []
for root, dirs, files in os.walk(data_path):
    for names in filter(lambda name: name[0] != '.', files):
        patient_list.append(os.path.join(root, names))
random.shuffle(patient_list)
'''


plt_test = check_slices_and_labels(patient_path, pos= [100, 100, 50], alpha = 1)
plt_test.show()