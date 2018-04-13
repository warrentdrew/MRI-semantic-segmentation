import numpy as np

import keras
import keras.backend as K


# losses
# ---------------------------------------------------------------------------

# jaccard_dist without background as class
def jaccard_dist(y_true, y_pred):           #might be that bachground is not added into the label
    smooth = 1e-12
    # amount of classes
    cl = K.shape(y_true)[-1]-1              #cl = 3 since bg is not considered
    # reshape
    y_true = K.reshape(y_true[...,1:], [-1, cl])
    y_pred = K.reshape(y_pred[...,1:], [-1, cl])
    # multiple-class-problem
    T = K.sum(y_true, axis=0)
    P = K.sum(K.square(y_pred), axis=0)
    PT = K.sum(y_pred * y_true, axis=0)
    denom = P + T - PT
    # average the jaccard-distance 
    jcd = (denom - PT + smooth) / (denom + smooth)
    cl = K.tf.to_float(cl, name='ToFloat')
    result = K.sum(jcd) / cl
    return result

# metrics
# ----------------------------------------------------------------------------

# accuracy without background
def my_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(y_true[..., 1:], y_pred[..., 1:])

# metric jaccard distance with discrete predictions
def jaccard_dist_discrete(y_true, y_pred):
    smooth = 1e-12
    # amount of classes
    cl = K.shape(y_true)[-1]-1
    # discrete, one hot predictions
    y_pred = K.one_hot(K.argmax(y_pred, axis=-1), cl+1)
    # reshape
    y_true = K.reshape(y_true[...,1:], [-1, cl])
    y_pred = K.reshape(y_pred[...,1:], [-1, cl])
    # multiple-class-problem
    T = K.sum(y_true, axis=0)
    P = K.sum(K.square(y_pred), axis=0)
    PT = K.sum(y_pred * y_true, axis=0)
    denom = P + T - PT
    # average the jaccard-distance 
    jcd = (denom - PT + smooth) / (denom + smooth)
    cl = K.tf.to_float(cl, name='ToFloat')
    result = K.sum(jcd) / cl
    return result

# functions for recall

# true-positives, numerator
def metric_tp_worker(c, y_true, y_pred):
    # only evaluate the argmax, actual result of classification
    cl = K.shape(y_true)[-1]            #cl = 4
    y_pred = K.one_hot(K.argmax(y_pred, axis=-1), cl)               #present the final result using a one-hot
    return K.sum((y_true[..., c] * y_pred[..., c]))                 #sum up the correctly classified pixels in each class

metric_tp = lambda c: lambda y_true, y_pred: metric_tp_worker(c, y_true, y_pred)

# ground-truth, denomiator
def metric_gt_worker(c, y_true, y_pred):
    return K.sum(y_true[..., c])

metric_gt = lambda c: lambda y_true, y_pred: metric_gt_worker(c, y_true, y_pred)
