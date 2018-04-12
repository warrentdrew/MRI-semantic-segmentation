import argparse
import os
# choose GPU
os.environ["CUDA_VISIBLE_DEVICES"]="3"

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

from dccn_re import DCCN_RE

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default= '/tmp/med_data_local/', type = str)
parser.add_argument('--model_path', default= './models/', type = str)
parser.add_argument('--history_path', default= './histories/', type = str)

parser.add_argument('--in_size', default = 32, type = int)
parser.add_argument('--k', default = 16, type = int, help= 'growth rate of dense block')
parser.add_argument('--ls', default = [8,8,8,12], type = list, help = 'layers in dense blocks')
parser.add_argument('--theta', default = 0.5, type = float, help = 'compression factor for dense net')
parser.add_argument('--k_0', default = 32, type = int, help = 'num of channel in input layer')
parser.add_argument('--lbda', default = 0, type = float, help = 'lambda for l2 reg')
parser.add_argument('--out_res', default=24, type = int, help = 'output resolution')
parser.add_argument('--with_pos', dest='pos', help = 'add position information in model', action = 'store_true')    #This is the same as mult_inputs
parser.add_argument('--no_pos', dest='pos', help = 'add position information in model', action = 'store_false')
parser.set_defaults(pos = True)
parser.add_argument('--pos_noise_stdv', default = 0, type = float, help = 'noise for position')
parser.add_argument('--epochs', default = 50, type = int)
parser.add_argument('--batch_size', default= 48, type= int)
parser.add_argument('--patient_buffer_capacity', default = 30, type=int)
parser.add_argument('--batches_per_shift', default = 25, type = int)
parser.add_argument('--density', default= 5, type = int)
parser.add_argument('--border', default= 20, type = int)
parser.add_argument('--empty_buffer', dest='empty', help = 'empty whole buffer after training of one model', action = 'store_true')
parser.add_argument('--no_empty_buffer', dest='empty', help = 'empty whole buffer after training of one model', action = 'store_false')
parser.set_defaults(empty = True)

args = parser.parse_args()

# paths to data (patients)
#path = "/tmp/med_data_local/"

# path were to save the model and history
#path_m = "/home/DCNet/models/"
#path_h = "/home/DCNet/histories/"

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
    patients_test, patients_train, patients_val, patients_val_slices = util.load_correct_patients(path=args.data_path,
                                                                                                  patients_to_take=45,                  # number of test patients
                                                                                                  forget_slices=True,                   # delete not needed loaded data on RAM
                                                                                                  cut=None,                             # split for validation- and training-data
                                                                                                  k=4,                                  # k-fold split
                                                                                                  perc=0.5,                             # sliding window overlap
                                                                                                  iteration=i,                          # k-fold cross-validation run
                                                                                                  last_val_patients=last_val_patients,
                                                                                                  verbose=True)
    # load model with parameters
    # receptive field: out_res < input_res
    # position: feed_pos = True
    print(' load model')
    #res = 32  # input resolution


    '''
    Model = models.DenseNet3D(in_shape=(args.in_size, args.in_size, args.in_size, 2),  # input shape
                              k=16,  # growth rate
                              ls=[8, 8, 8, 12],  # layers in dense blocks
                              theta=0.5,  # compression factor
                              k_0=32,  # number of channels in input layer
                              lbda=0,  # optional weight-decay
                              out_res=24,  # receptive field: out_res < in_res
                              feed_pos=True,  # add position at bottleneck
                              pos_noise_stdv=0)  # optional noise for position
    '''
    network = DCCN_RE(in_shape=(args.in_size, args.in_size, args.in_size, 2),
                      k = args.k,
                      ls = args.ls,
                      thera = args.theta,
                      k_0 = args.k_0,
                      lbda = args.lbda,
                      out_res= args.out_res,
                      feed_pos= args.pos,
                      pos_noise_stdv = args.pos_noise_stdv)

    # compile model
    # settings for true-positive-rate (TPR)
    '''
    cls = 3

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
    '''

    network.compile()

    # saves the model weights after each epoch if the validation loss decreased
    path_w = args.model_path + "k-fold-50" + str(i) + ".hdf5"
    checkpointer = cb.ModelCheckpoint(filepath=path_w, verbose=0, monitor='val_loss', save_best_only=True)

    # train
    # postion: mult_inputs = True
    hist_object = network.train(patients_train, patients_val_slices, checkpointer, args)

    print(' save histories')
    # list of histories
    lhist.append(hist_object.history)

    # save history
    path_hist = args.history_path + "k-fold-50"
    history.save_histories(lhist=lhist, path=path_hist)