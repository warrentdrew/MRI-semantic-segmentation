import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import matplotlib.pyplot as plt
import numpy as np
from postprocess.history import lhist_to_dictarr


# plot jaccard distance curve when training with focal loss
def plot_jaccard_distance_curve(lhist, ep):
    plt.figure(figsize=(9, 5))
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    dictarr = lhist_to_dictarr(lhist)

    train_loss = dictarr['seg_out_jaccard_dist_discrete']
    # val_loss = dictarr['val_seg_out_jaccard_dist']
    val_discr_loss = dictarr['val_seg_out_jaccard_dist_discrete']

    epochs = list(range(1, ep + 1))
    plt.xticks(list(range(1, ep + 2, 3)))

    train_loss_mean = np.mean(train_loss, axis=0)
    train_loss_std = np.std(train_loss, axis=0)

    # val_loss_mean = np.mean(val_loss, axis=0)
    # val_loss_std = np.std(val_loss, axis=0)

    val_discr_loss_mean = np.mean(val_discr_loss, axis=0)
    val_discr_loss_std = np.std(val_discr_loss, axis=0)

    plt.ylim(*(0, 0.07))
    plt.yticks(np.linspace(0, 0.7, 11))
    plt.grid()

    plt.fill_between(epochs, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, alpha=0.1, color="g")
    # plt.fill_between(epochs, val_loss_mean - val_loss_std, val_loss_mean + val_loss_std, alpha=0.1, color="r")
    plt.fill_between(epochs, val_discr_loss_mean - val_discr_loss_std, val_discr_loss_mean + val_discr_loss_std,
                     alpha=0.1, color="b")

    plt.plot(epochs, train_loss_mean, '-', color="g", label="training loss")
    # plt.plot(epochs, val_loss_mean, '-', color="r", label="validation loss")
    plt.plot(epochs, val_discr_loss_mean, '-', color="b", label="discrete validation loss")

    plt.legend(loc="upper right")

    return plt


# plot trainings recall curve
def plot_train_recall_curve(lhist, ep):
    plt.figure(figsize=(9, 5))
    plt.title("Train Recall Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")

    dictarr = lhist_to_dictarr(lhist)

    m_tp_c0 = dictarr['m_tp_c0']
    m_tp_c1 = dictarr['m_tp_c1']
    m_tp_c2 = dictarr['m_tp_c2']
    m_tp_c3 = dictarr['m_tp_c3']

    m_gt_c0 = dictarr['m_gt_c0']
    m_gt_c1 = dictarr['m_gt_c1']
    m_gt_c2 = dictarr['m_gt_c2']
    m_gt_c3 = dictarr['m_gt_c3']

    epochs = list(range(1, ep + 1))
    plt.xticks(list(range(1, ep + 2, 3)))

    recall_c0 = np.divide(m_tp_c0, m_gt_c0)
    recall_c1 = np.divide(m_tp_c1, m_gt_c1)
    recall_c2 = np.divide(m_tp_c2, m_gt_c2)
    recall_c3 = np.divide(m_tp_c3, m_gt_c3)
    '''


    recall_c0 = dictarr['seg_out_recall_bg']
    recall_c1 = dictarr['seg_out_recall_lt']
    recall_c2 = dictarr['seg_out_recall_vat']
    recall_c3 = dictarr['seg_out_recall_scat']

    '''

    recall_c0_mean = np.mean(recall_c0, axis=0)
    recall_c1_mean = np.mean(recall_c1, axis=0)
    recall_c2_mean = np.mean(recall_c2, axis=0)
    recall_c3_mean = np.mean(recall_c3, axis=0)

    recall_c0_std = np.std(recall_c0, axis=0)
    recall_c1_std = np.std(recall_c1, axis=0)
    recall_c2_std = np.std(recall_c2, axis=0)
    recall_c3_std = np.std(recall_c3, axis=0)

    plt.ylim(*(0.6, 1.01))
    plt.yticks(np.linspace(0.6, 1, 11))
    plt.grid()

    plt.fill_between(epochs, (recall_c0_mean - recall_c0_std), (recall_c0_mean + recall_c0_std), alpha=0.1,
                     color="black")
    plt.fill_between(epochs, (recall_c1_mean - recall_c1_std), (recall_c1_mean + recall_c1_std), alpha=0.1, color="r")
    plt.fill_between(epochs, (recall_c2_mean - recall_c2_std), (recall_c2_mean + recall_c2_std), alpha=0.1, color="g")
    plt.fill_between(epochs, (recall_c3_mean - recall_c3_std), (recall_c3_mean + recall_c3_std), alpha=0.1, color="b")

    plt.plot(epochs, recall_c0_mean, '-', color="black", label="recall class 0: background")
    plt.plot(epochs, recall_c1_mean, '-', color="r", label="recall class 1: lean tissue")
    plt.plot(epochs, recall_c2_mean, '-', color="g", label="recall class 2: visceral AT")
    plt.plot(epochs, recall_c3_mean, '-', color="b", label="recall class 3: subcutaneous AT")
    plt.legend(loc="lower right")

    return plt


# plot validation recall curve
def plot_val_recall_curve(lhist, ep):
    plt.figure(figsize=(9, 5))
    plt.title("Validation: Recall Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")

    dictarr = lhist_to_dictarr(lhist)

    m_tp_c0 = dictarr['val_m_tp_c0']
    m_tp_c1 = dictarr['val_m_tp_c1']
    m_tp_c2 = dictarr['val_m_tp_c2']
    m_tp_c3 = dictarr['val_m_tp_c3']

    m_gt_c0 = dictarr['val_m_gt_c0']
    m_gt_c1 = dictarr['val_m_gt_c1']
    m_gt_c2 = dictarr['val_m_gt_c2']
    m_gt_c3 = dictarr['val_m_gt_c3']

    epochs = list(range(1, ep + 1))
    plt.xticks(list(range(1, ep + 2, 3)))

    recall_c0 = np.divide(m_tp_c0, m_gt_c0)
    recall_c1 = np.divide(m_tp_c1, m_gt_c1)
    recall_c2 = np.divide(m_tp_c2, m_gt_c2)
    recall_c3 = np.divide(m_tp_c3, m_gt_c3)

    '''

    recall_c0 = dictarr['val_seg_out_recall_bg']
    recall_c1 = dictarr['val_seg_out_recall_lt']
    recall_c2 = dictarr['val_seg_out_recall_vat']
    recall_c3 = dictarr['val_seg_out_recall_scat']
    '''

    recall_c0_mean = np.mean(recall_c0, axis=0)
    recall_c1_mean = np.mean(recall_c1, axis=0)
    recall_c2_mean = np.mean(recall_c2, axis=0)
    recall_c3_mean = np.mean(recall_c3, axis=0)

    recall_c0_std = np.std(recall_c0, axis=0)
    recall_c1_std = np.std(recall_c1, axis=0)
    recall_c2_std = np.std(recall_c2, axis=0)
    recall_c3_std = np.std(recall_c3, axis=0)

    plt.ylim(*(0.6, 1.01))
    plt.yticks(np.linspace(0.6, 1, 11))
    plt.grid()

    plt.fill_between(epochs, (recall_c0_mean - recall_c0_std), (recall_c0_mean + recall_c0_std), alpha=0.1,
                     color="black")
    plt.fill_between(epochs, (recall_c1_mean - recall_c1_std), (recall_c1_mean + recall_c1_std), alpha=0.1, color="r")
    plt.fill_between(epochs, (recall_c2_mean - recall_c2_std), (recall_c2_mean + recall_c2_std), alpha=0.1, color="g")
    plt.fill_between(epochs, (recall_c3_mean - recall_c3_std), (recall_c3_mean + recall_c3_std), alpha=0.1, color="b")

    plt.plot(epochs, recall_c0_mean, '-', color="black", label="recall class 0: background")
    plt.plot(epochs, recall_c1_mean, '-', color="r", label="recall class 1: lean tissue")
    plt.plot(epochs, recall_c2_mean, '-', color="g", label="recall class 2: visceral AT")
    plt.plot(epochs, recall_c3_mean, '-', color="b", label="recall class 3: subcutaneous AT")
    plt.legend(loc="lower right")

    return plt