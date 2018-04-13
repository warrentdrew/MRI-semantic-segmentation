# Example Training Script
# ----------------------------------------------------
# versions: python 3.x, keras 2.0.5, tensorflow 1.3.0

# one epoch with the parameters below and
# over all available data (all 173 patients)
# will last about 20 minutes
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)



#import os

# choose GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import keras
import keras.backend as K
from keras import callbacks as cb
from keras.models import Sequential, Model
from keras.layers import *

K.set_image_data_format = 'channels_last'

# import own scripts
#import libs.util as util
#import libs.preprocessing as preprocessing
#from libs.training import *

import libs.custom_metrics as custom_metrics
import dccn_ori as models
import libs.history as history
from utils import dataset_split
from train import fit
# paths to data (patients)
path = "../patient-paths/patients_1_5T.pkl"

# path were to save the model and history
path_m = "../models/"
path_h = "../histories/"

# list for history-objects
lhist = []

# train model 4 times with k-fold cross validation

for i in range(4):

    print(' round ' + str(i) + '!')
    print(' load patients and drop last validation patients')

    #do the data loading and preprocessing
    train_path, validation_path, test_path = dataset_split(path, test_rate = 0.2, valid_train_rate = 0.1, shuffle = True, seed = 100)

    # load model with parameters
    # receptive field: out_res < input_res
    # position: feed_pos = True
    print(' load model')
    res = 32  # input resolution
    Model = models.DenseNet3D(in_shape=(res, res, res, 1),  # input shape
                              k=16,  # growth rate
                              ls=[8, 8, 8, 12],  # layers in dense blocks
                              theta=0.5,  # compression factor
                              k_0=32,  # number of channels in input layer
                              lbda=0,  # optional weight-decay
                              out_res=None,  # receptive field: out_res < in_res
                              feed_pos=False,  # add position at bottleneck
                              pos_noise_stdv=0)  # optional noise for position

    # compile model
    # settings for true-positive-rate (TPR)
    cls = 4

    m1 = [custom_metrics.metric_tp(c) for c in range(cls)]
    for j, f in enumerate(m1):
        f.__name__ = 'm_tp_c' + str(j)

    m2 = [custom_metrics.metric_gt(c) for c in range(cls)]
    for k, f in enumerate(m2):
        f.__name__ = 'm_gt_c' + str(k)

    Model.compile(optimizer='rmsprop',
                  loss=custom_metrics.jaccard_dist,
                  metrics=m1 + m2 + ['categorical_accuracy'] + [custom_metrics.jaccard_dist_discrete])
    print(' model compiled.')

    # saves the model weights after each epoch if the validation loss decreased
    path_w = path_m + "k-fold-50" + str(i) + ".hdf5"
    checkpointer = cb.ModelCheckpoint(filepath=path_w, verbose=0, monitor='val_loss', save_best_only=True)

    # train
    # postion: mult_inputs = True
    print(' training...')
    hist_object = fit(model = Model,
                      patients_train = train_path,
                      patients_valid = validation_path,
                      epochs = 50,
                      batch_size = 64,
                      patient_buffer_capacity = 16,
                      batches_per_shift = 32,
                      steps_valid = 128,
                      border = 20,
                      callbacks = [checkpointer],
                      mult_inputs=False)

    print(' save histories')
    # list of histories
    lhist.append(hist_object.history)

    # save history
    path_hist = path_h + "k-fold-50"
    history.save_histories(lhist=lhist, path=path_hist)
