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

def plot_loss_curve(lhist):
    
    plt.figure(figsize=(9,5))
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    dictarr = lhist_to_dictarr(lhist)
 
    train_loss = dictarr['loss']
    val_loss = dictarr['val_loss']

    if len(train_loss.shape) > 1:
        ep = train_loss.shape[1]
    else: ep = train_loss.shape

    epochs = list(range(1,ep+1))

    train_loss_mean = np.mean(train_loss, axis=0)
    train_loss_std = np.std(train_loss, axis=0)

    val_loss_mean = np.mean(val_loss, axis=0)
    val_loss_std = np.std(val_loss, axis=0)
    
    plt.grid()
    plt.xlim(1,ep)
    plt.ylim(0,1)
    
    plt.fill_between(epochs, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, alpha=0.1, color="g")
    plt.fill_between(epochs, val_loss_mean - val_loss_std, val_loss_mean + val_loss_std, alpha=0.1, color="r")
    
    plt.plot(epochs, train_loss_mean, '-', color="g", label="training loss")
    plt.plot(epochs, val_loss_mean, '-', color="r", label="validation loss")
    
    plt.legend(loc="upper right")
    
    return plt

def plot_train_recall_curve(lhist, m1, m2):
    
    plt.figure(figsize=(9,5))
    plt.title("Train Recall Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    
    dictarr = lhist_to_dictarr(lhist)  

    tp_names = [f.__name__ for f in m1]
    gt_names = [f.__name__ for f in m2]
 
    tps = [dictarr[name] for name in tp_names]
    gts = [dictarr[name] for name in gt_names]

    if len(tps[0].shape) > 1:
        ep = tps[0].shape[1]
    else: ep = tps[0].shape
        
    epochs = list(range(1,ep+1))

    recalls = np.divide(tps, gts)
    recalls_mean = np.mean(recalls, axis=1)
    recalls_std = np.std(recalls, axis=1)
 
    plt.grid()
    plt.xlim(1,ep)
    plt.ylim(0,1+0.05)

    for i in range(recalls.shape[0]):
        plt.fill_between(epochs, (recalls_mean[i] - recalls_std[i]), (recalls_mean[i] + recalls_std[i]), alpha=0.1)
        plt.plot(epochs, recalls_mean[i], '-', label='recall class ' + str(i))
    
    plt.legend(loc="lower right")
    
    return plt

def plot_val_recall_curve(lhist, m1, m2):
    
    plt.figure(figsize=(9,5))
    plt.title("Validation: Recall Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    
    dictarr = lhist_to_dictarr(lhist)  

    tp_names = ['val_' + f.__name__ for f in m1]
    gt_names = ['val_' + f.__name__ for f in m2]
 
    tps = [dictarr[name] for name in tp_names]
    gts = [dictarr[name] for name in gt_names]

    if len(tps[0].shape) > 1:
        ep = tps[0].shape[1]
    else: ep = tps[0].shape
        
    epochs = list(range(1,ep+1))

    recalls = np.divide(tps, gts)
    recalls_mean = np.mean(recalls, axis=1)
    recalls_std = np.std(recalls, axis=1)
    
    plt.grid()
    plt.xlim(1,ep)
    plt.ylim(0,1+0.05)

    for i in range(recalls.shape[0]):
        plt.fill_between(epochs, (recalls_mean[i] - recalls_std[i]), (recalls_mean[i] + recalls_std[i]), alpha=0.1)
        plt.plot(epochs, recalls_mean[i], '-', label='recall class ' + str(i))
    
    plt.legend(loc="lower right")
    
    return plt
