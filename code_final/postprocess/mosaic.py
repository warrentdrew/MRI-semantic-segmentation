import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import matplotlib.pyplot as plt
import math
import numpy as np
from util.util_at import get_num_mat, colorize, show_slice_num

def save_mosaic_style(patient, model, model_name, col, border, add_pos, add_prh,
                      space=0, back=0, save_rt='/home/d1251/Downloads/'):

    img, label = patient.get_slices(count=False)  # img shape x,y,z,1
    (h, w, slices, _) = img.shape
    label = np.argmax(label, axis=-1)  # shape is x,y,1,1

    prediction = patient.predict_patient(model, border, add_pos, add_prh)
    prediction = np.argmax(prediction, axis=-1)

    # compute num of rows and cols and spaces
    row = math.ceil(slices / col)
    slice_idx = 0
    num_mat = get_num_mat('/home/d1251/no_backup/d1251/aux_file/numbers.mat')

    for i in range(row):
        for j in range(col):
            if slice_idx < slices:
                f_img = img[:, :, slice_idx].reshape([h, w, 1])
                f_label = colorize(label[:, :, slice_idx])
                f_pred = colorize(prediction[:, :, slice_idx])
                f_img = show_slice_num(f_img, num_mat, slice_idx + 1).reshape([h, w])
                f_label = show_slice_num(f_label, num_mat, slice_idx + 1)
                f_pred = show_slice_num(f_pred, num_mat, slice_idx + 1)
            else:
                f_img = np.zeros([h, w])
                f_label = f_pred = np.zeros([h, w, 3])
            if j == 0:
                R_img = f_img
                R_label = f_label
                R_pred = f_pred
            else:
                if space is not 0:
                    vspace = back * (np.ones([h, space]))  # vspace stands for the vertical space on columns
                    vspace_3c = back * (np.ones([h, space, 3]))  # vspace stands for the vertical space on columns
                    R_img = np.concatenate([R_img, vspace, f_img], axis=1)
                    R_label = np.concatenate([R_label, vspace_3c, f_label], axis=1)
                    R_pred = np.concatenate([R_pred, vspace_3c, f_pred], axis=1)

                else:
                    R_img = np.concatenate([R_img, f_img], axis=1)
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

    patient_id = patient.get_patient_id().replace('/', '_')
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