import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import argparse
import os
# choose GPU
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import keras
import keras.backend as K
from keras import callbacks as cb
from keras.models import Sequential, Model
from keras.layers import *
K.set_image_data_format = 'channels_last'

import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
#config = K.tf.ConfigProto(intra_op_parallelism_threads = 1, inter_op_parallelism_threads = 1, allow_soft_placement = True, device_count = {'CPU': 1})
#config.gpu_options.per_process_gpu_memory_fraction = 0.95
#config.intra_op_parallelism_threads=1
#config.inter_op_parallelism_threads=1
#K.set_session(tf.Session(config=config))

# import own scripts

import libs.history as history
from mrge_pr import MRGE_PR
from utils import dataset_split, load_patient_paths
from patient import load_correct_patient



parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default= '../patient-paths/patients_1_5T.pkl', type = str)
parser.add_argument('--model_path', default= '/home/d1251/no_backup/d1251/models/', type = str)
parser.add_argument('--history_path', default= '/home/d1251/no_backup/d1251/histories/', type = str)
parser.add_argument('--in_size', default = 32, type = int)
parser.add_argument('--rls', default = [8,4,2,1,1], type = list, help = 'layers in dense blocks')
parser.add_argument('--k_0', default = 32, type = int, help = 'num of channel in input layer')
parser.add_argument('--lbda', default = 0, type = float, help = 'lambda for l2 reg')
parser.add_argument('--out_res', default=None, type = int, help = 'output resolution')
parser.add_argument('--with_pos', dest='pos', help = 'add position information in model', action = 'store_true')    #This is the same as mult_inputs
parser.add_argument('--no_pos', dest='pos', help = 'add position information in model', action = 'store_false')
parser.set_defaults(pos = True)
parser.add_argument('--pos_noise_stdv', default = 0, type = float, help = 'noise for position')
parser.add_argument('--epochs', default = 60, type = int)
parser.add_argument('--batch_size', default= 16, type= int)
parser.add_argument('--patient_buffer_capacity', default = 10, type=int)
parser.add_argument('--batches_per_shift', default = 15, type = int)
parser.add_argument('--density', default= 5, type = int)
parser.add_argument('--border', default= 5, type = int)
parser.add_argument('--empty_buffer', dest='empty', help = 'empty whole buffer after training of one model', action = 'store_true')
parser.add_argument('--no_empty_buffer', dest='empty', help = 'empty whole buffer after training of one model', action = 'store_false')
parser.set_defaults(empty = True)

args = parser.parse_args()



# list for history-objects
lhist = []

#load patients list for the training
patients_path_list = load_patient_paths(args.data_path, shuffle = True, seed = 100)

# train model 4 times with k-fold cross validation
k = 4
for i in range(4):

    print(' round ' + str(i) + '!')
    print(' load patients and drop last validation patients')

    # do the data loading and preprocessing
    train_path, validation_path, test_path = dataset_split(patients_path_list, k, i)
    patients_train, patients_val, patients_test, patients_val_slices = load_correct_patient(train_path, validation_path,
                                                                                            test_path,
                                                                                            forget_slices=True)

    # load model with parameters
    # receptive field: out_res < input_res
    # position: feed_pos = True
    print(' load model')
    #res = 32  # input resolution



    network = MRGE_PR(in_shape=(args.in_size, args.in_size, args.in_size, 1),
                      rls = args.rls,
                      k_0 = args.k_0,
                      lbda = args.lbda,
                      out_res= args.out_res,
                      feed_pos= args.pos,
                      pos_noise_stdv = args.pos_noise_stdv)

    #compile model
    network.compile()

    network.model.summary()

    # saves the model weights after each epoch if the validation loss decreased
    path_w = args.model_path + "k-fold-mrge-pr-weights" + str(i) + ".hdf5"
    checkpointer = cb.ModelCheckpoint(filepath=path_w, verbose=0, monitor='val_loss', save_best_only=True, save_weights_only=True)

    # train
    # postion: mult_inputs = True
    hist_object = network.train(patients_train, patients_val_slices, checkpointer, args)

    print(' save histories')
    # list of histories
    lhist.append(hist_object.history)

    # save history
    path_hist = args.history_path + "k-fold-mrge-pr"
    history.save_histories(lhist=lhist, path=path_hist)