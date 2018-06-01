import numpy as np
import pickle

import matplotlib.pyplot as plt


def save_histories(lhist, path):
    with open(path, 'wb') as f:
        pickle.dump(lhist,f)

def load_histories(path):
    with open(path, 'rb') as f:
        lhist = pickle.load(f)
    return lhist

def lhist_to_dictarr(lhist):
    dictarr = {}
    l = []
    keys = lhist[0].keys()
    for key in keys:
        for hist in lhist:
            l.append(hist[key])
        dictarr[key] = np.array(l)
        l = []
    return dictarr


# plot trainings recall curve
def plot_train_recall_curve(lhist, ep):
    
    plt.figure(figsize=(9,5))
    plt.title("Train Recall Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    
    dictarr = lhist_to_dictarr(lhist)
    
    m_tp_c0 = dictarr['m_tp_c0']
    m_tp_c1 = dictarr['m_tp_c1']
    m_tp_c2 = dictarr['m_tp_c2']

    m_gt_c0 = dictarr['m_gt_c0']
    m_gt_c1 = dictarr['m_gt_c1']
    m_gt_c2 = dictarr['m_gt_c2']
        
    epochs = list(range(1,ep+1))
    plt.xticks(list(range(1,ep+2,3)))

    recall_c0 = np.divide(m_tp_c0, m_gt_c0)
    recall_c1 = np.divide(m_tp_c1, m_gt_c1)
    recall_c2 = np.divide(m_tp_c2, m_gt_c2)

    recall_c0_mean = np.mean(recall_c0, axis=0)
    recall_c1_mean = np.mean(recall_c1, axis=0)
    recall_c2_mean = np.mean(recall_c2, axis=0)

    recall_c0_std = np.std(recall_c0, axis=0)
    recall_c1_std = np.std(recall_c1, axis=0)
    recall_c2_std = np.std(recall_c2, axis=0)

    
    plt.ylim(*(0.6,1.01))
    plt.yticks(np.linspace(0.6, 1, 11))
    plt.grid()

    plt.fill_between(epochs, (recall_c0_mean - recall_c0_std), (recall_c0_mean + recall_c0_std), alpha=0.1, color="black")
    plt.fill_between(epochs, (recall_c1_mean - recall_c1_std), (recall_c1_mean + recall_c1_std), alpha=0.1, color="r")
    plt.fill_between(epochs, (recall_c2_mean - recall_c2_std), (recall_c2_mean + recall_c2_std), alpha=0.1, color="g")

    plt.plot(epochs, recall_c0_mean, '-', color="black", label="recall class 0: background")
    plt.plot(epochs, recall_c1_mean, '-', color="r", label="recall class 1: liver")
    plt.plot(epochs, recall_c2_mean, '-', color="g", label="recall class 2: spleen")
    plt.legend(loc="lower right")
    
    return plt

# plot validation recall curve
def plot_val_recall_curve(lhist, ep):
    
    plt.figure(figsize=(9,5))
    plt.title("Validation: Recall Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    
    dictarr = lhist_to_dictarr(lhist)
    
    m_tp_c0 = dictarr['val_m_tp_c0']
    m_tp_c1 = dictarr['val_m_tp_c1']
    m_tp_c2 = dictarr['val_m_tp_c2']

    m_gt_c0 = dictarr['val_m_gt_c0']
    m_gt_c1 = dictarr['val_m_gt_c1']
    m_gt_c2 = dictarr['val_m_gt_c2']
        
    epochs = list(range(1,ep+1))
    plt.xticks(list(range(1,ep+2,4)))

    recall_c0 = np.divide(m_tp_c0, m_gt_c0)
    recall_c1 = np.divide(m_tp_c1, m_gt_c1)
    recall_c2 = np.divide(m_tp_c2, m_gt_c2)

    recall_c0_mean = np.mean(recall_c0, axis=0)
    recall_c1_mean = np.mean(recall_c1, axis=0)
    recall_c2_mean = np.mean(recall_c2, axis=0)

    recall_c0_std = np.std(recall_c0, axis=0)
    recall_c1_std = np.std(recall_c1, axis=0)
    recall_c2_std = np.std(recall_c2, axis=0)
    
    plt.ylim(*(0.6,1.01))
    plt.yticks(np.linspace(0.6, 1, 11))
    plt.grid()

    plt.fill_between(epochs, (recall_c0_mean - recall_c0_std), (recall_c0_mean + recall_c0_std), alpha=0.1, color="black")
    plt.fill_between(epochs, (recall_c1_mean - recall_c1_std), (recall_c1_mean + recall_c1_std), alpha=0.1, color="r")
    plt.fill_between(epochs, (recall_c2_mean - recall_c2_std), (recall_c2_mean + recall_c2_std), alpha=0.1, color="g")

    plt.plot(epochs, recall_c0_mean, '-', color="black", label="recall class 0: background")
    plt.plot(epochs, recall_c1_mean, '-', color="r", label="recall class 1: liver")
    plt.plot(epochs, recall_c2_mean, '-', color="g", label="recall class 2: spleen")
    plt.legend(loc="lower right")
    
    return plt

# plot loss curve
def plot_loss_curve(lhist, ep):
    
    plt.figure(figsize=(9,5))
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    dictarr = lhist_to_dictarr(lhist)
 
    train_loss = dictarr['loss']
    val_loss = dictarr['val_loss']
    val_discr_loss = dictarr['val_jaccard_dist_discrete']

    epochs = list(range(1,ep+1))
    plt.xticks(list(range(1,ep+2,3)))

    train_loss_mean = np.mean(train_loss, axis=0)
    train_loss_std = np.std(train_loss, axis=0)

    val_loss_mean = np.mean(val_loss, axis=0)
    val_loss_std = np.std(val_loss, axis=0)

    val_discr_loss_mean = np.mean(val_discr_loss, axis=0)
    val_discr_loss_std = np.std(val_discr_loss, axis=0)
    
    plt.ylim(*(0,0.07))
    plt.yticks(np.linspace(0, 0.7, 11))
    plt.grid()
    
    plt.fill_between(epochs, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, alpha=0.1, color="g")
    plt.fill_between(epochs, val_loss_mean - val_loss_std, val_loss_mean + val_loss_std, alpha=0.1, color="r")
    plt.fill_between(epochs, val_discr_loss_mean - val_discr_loss_std, val_discr_loss_mean + val_discr_loss_std, alpha=0.1, color="b")
    
    plt.plot(epochs, train_loss_mean, '-', color="g", label="training loss")
    plt.plot(epochs, val_loss_mean, '-', color="r", label="validation loss")
    plt.plot(epochs, val_discr_loss_mean, '-', color="b", label="discrete validation loss")
    
    plt.legend(loc="upper right")

    
    return plt
