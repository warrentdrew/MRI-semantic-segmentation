# Example Training Script 
#----------------------------------------------------
# versions: python 3.x, keras 2.0.5, tensorflow 1.3.0

# one epoch with the parameters below and
# over all available data (all 173 patients) 
# will last about 20 minutes

import os

# choose GPU
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import keras
import keras.backend as K
from keras import callbacks as cb
from keras.models import Sequential, Model
from keras.layers import *
K.set_image_data_format = 'channels_last'

# import own scripts
import libs.util
import libs.preprocessing
from libs.training import *
import libs.custom_metrics
import libs.models
import libs.history


# paths to data (patients)
path="/tmp/med_data_local/"

# path were to save the model and history
path_m = "/home/DCNet/models/" 
path_h = "/home/DCNet/histories/"      

# list for history-objects
lhist = []

# train model 4 times with k-fold cross validation

for i in range(4):

    print(' round ' + str(i) + '!')
    print(' load patients and drop last validation patients')

    if i > 0:
        last_val_patients = patients_val
    else:
        last_val_patients = None

    # k=4, 50% out of every 1/4-part, 45 patients discarded (as test-patients)     #why 50 % out of every 1/4 part, what does sliding window overlap mean???
    patients_test, patients_train, patients_val, patients_val_slices = util.load_correct_patients(path=path, 
                                                                                                  patients_to_take=45, 	# number of test patients
                                                                                                  forget_slices=True, 	# delete not needed loaded data on RAM
                                                                                                  cut=None, 		    # split for validation- and training-data
                                                                                                  k=4, 			        # k-fold split
                                                                                                  perc=0.5,		        # sliding window overlap
                                                                                                  iteration=i,		    # k-fold cross-validation run
                                                                                                  last_val_patients=last_val_patients, 
                                                                                                  verbose=True)
    # load model with parameters 
    # receptive field: out_res < input_res
    # position: feed_pos = True
    print(' load model')
    res = 32            #input resolution
    Model = models.DenseNet3D(in_shape=(res,res,res,2),		# input shape
                              k=16,				            # growth rate
                              ls=[8,8,8,12],			    # layers in dense blocks
                              theta=0.5,			        # compression factor
                              k_0=32,				        # number of channels in input layer
                              lbda=0,                       # optional weight-decay
                              out_res=24,                   # receptive field: out_res < in_res
                              feed_pos=True,                # add position at bottleneck
                              pos_noise_stdv=0)             # optional noise for position
        
    # compile model
    # settings for true-positive-rate (TPR)
    cls = 3

    m1 = [custom_metrics.metric_tp(c) for c in range(cls)]
    for j, f in enumerate(m1):
        f.__name__ = 'm_tp_c' + str(j)

    m2 = [custom_metrics.metric_gt(c) for c in range(cls)]
    for k, f in enumerate(m2):
        f.__name__ = 'm_gt_c' + str(k)

    Model.compile(optimizer='rmsprop',
                  loss=custom_metrics.jaccard_dist,
                  metrics = m1 + m2 + ['categorical_accuracy'] + [custom_metrics.jaccard_dist_discrete])
    print(' model compiled.')

    # saves the model weights after each epoch if the validation loss decreased
    path_w = path_m + "k-fold-50" + str(i) + ".hdf5"
    checkpointer = cb.ModelCheckpoint(filepath=path_w, verbose=0, monitor='val_loss', save_best_only=True)

    # train
    # postion: mult_inputs = True
    print(' training...')
    hist_object = fit(model=Model,
                      patients_train=patients_train,
                      data_valid=patients_val_slices,
                      epochs=50,
                      batch_size=48,
                      patient_buffer_capacity=30,	    # amount of patients on RAM
                      batches_per_shift=25,		        # batches_per_train_epoch = batches_per_shift * len(patients_train), 
                                                        # batches out of buffer before one shift-operation, see every patient in one epoch!
                      density=5,			            # density for meshgrid of positions for validation data
                      border=20,			            # distance in pixel between crops
                      callbacks=[checkpointer],		    # callback (see keras documentation) for validation loss
                      mult_inputs=True,			        # if additional position at bottleneck, mult_inputs = True
                      empty_patient_buffer=True)	    # empty whole buffer, after training of one model (provide RAM for next model)

    print(' save histories')
    # list of histories
    lhist.append(hist_object.history)

    # save history
    path_hist = path_h + "k-fold-50"
    history.save_histories(lhist=lhist, path=path_hist)
