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
K.set_image_dim_ordering('tf')

import tensorflow as tf
'''
config = K.tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9        #set memory amount used in each GPU
K.set_session(tf.Session(config=config))
'''

# import own scripts
import postprocess.history as history
from network.MRGE import MRGE
from util.util_at import dataset_split, load_patient_paths, get_3t_val_paths, load_correct_patient, Patient_AT
#for case == result
from postprocess.history import load_histories, plot_loss_curve
from postprocess.history_at import plot_train_recall_curve, plot_val_recall_curve
from postprocess.export import export_result
from postprocess.mosaic import save_mosaic_style
import random
import yaml

def train(cf):
    # list for history-objects
    lhist = []

    #load patients list for the training
    patients_path_list = load_patient_paths(cf['Paths']['data'], shuffle = True, seed = cf['CV']['seed'])

    # train model 4 times with k-fold cross validation
    k = cf['CV']['k']

    for i in range(k):

        print(' round ' + str(i) + '!')
        print(' load patients and drop last validation patients')

        if i > 0:
            last_val_patients = patients_val
        else:
            last_val_patients = None

        # do the data loading and preprocessing
        train_path, validation_path, test_path = dataset_split(patients_path_list, k, i, last_val_patients, train_num = 150, seed=100)
        for path in validation_path:
            print(path)

        patients_train, patients_val, patients_test, patients_val_slices = load_correct_patient(train_path, validation_path,
                                                                                                test_path,
                                                                                                forget_slices=True)
        # load model with parameters
        # receptive field: out_res < input_res
        # position: feed_pos = True
        print(' load model')

        network = MRGE(in_shape=(cf['Preprocess']['in_size'], cf['Preprocess']['in_size'], cf['Preprocess']['in_size'], 1),
                          rls = cf['Model']['rls'],
                          k_0 = cf['Model']['k0'],
                          multi = cf['Model']['multi'],
                          lbda = cf['Model']['lbda'],
                          out_res= cf['Preprocess']['out_res'],
                          feed_pos= cf['Model']['feed_pos'],
                          pos_noise_stdv = cf['Model']['pos_noise_stdv'])

        #compile model
        network.compile()
        network.model.summary()

        # saves the model weights after each epoch if the validation loss decreased
        path_w = cf['Paths']['model'] + "mrge_test" + str(i) + ".hdf5"
        checkpointer = cb.ModelCheckpoint(filepath=path_w, verbose=0, monitor='val_loss', save_best_only=True, save_weights_only=True)

        # train
        hist_object = network.train(patients_train, patients_val_slices, checkpointer, cf)
        #hist_object = network.resume(patients_train, patients_val_slices, [checkpointer], args, init_epoch = 0, model_path = '/home/d1251/no_backup/d1251/models/mrge-focal-0.hdf5')
        print(' save histories')
        # list of histories
        lhist.append(hist_object.history)

        # save history
        path_hist = cf['Paths']['histories'] + "mrge_test"
        history.save_histories(lhist=lhist, path=path_hist)

def test(cf):
    patient_paths = load_patient_paths(load_path=cf['Paths']['data'], shuffle=True, seed=100)
    mrge_model_name = 'mrge-focal-0.hdf5'
    mrge_model_path = os.path.join(cf['Paths']['model'], mrge_model_name)

    mrge_pos = MRGE(in_shape=(32, 32, 32, 1), rls=[8, 4, 2, 1, 1], k_0=16, feed_pos=True)

    K.get_session().run(tf.global_variables_initializer())
    mrge_pos.model.load_weights(mrge_model_path)
    random.seed(100)

    for i in range(10):
        patient_path = random.choice(patient_paths)
        print(patient_path)
        patient = Patient_AT(patient_path, forget_slices=True)
        save_mosaic_style(patient, model = mrge_pos.model, model_name = 'focal_mrge', add_pos= True, add_prh = False, col = 5, border = 20,
                                  space = 1, back = 1, save_rt = '/home/d1251/no_backup/d1251/results/mrge/prediction/3T_mosaic')



def result(cf):
    lhist = load_histories('/home/d1251/no_backup/d1251/histories/3-fold-60-ori')
    plt_loss = plot_loss_curve(lhist=lhist, ep=60)
    plt_train_recall = plot_train_recall_curve(lhist=lhist, ep=60)
    plt_val_recall = plot_val_recall_curve(lhist=lhist, ep=60)

    plt_loss.show()
    plt_train_recall.show()
    plt_val_recall.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DenseNet training')

    parser.add_argument('-c', '--config_path',
                        type=str,
                        default='config/config.yml',
                        help='Configuration file')

    parser.add_argument('-e', '--exp_name',
                        type=str,
                        default=None,
                        help='Name of experiment')

    arguments = parser.parse_args()

    assert arguments.config_path is not None, 'Please provide a configuration path using' \
                                              ' -c pathname in the command line.'
    assert arguments.exp_name is not None, 'Please provide a name for the experiment' \
                                           ' -e name in the command line'

    # Parse the configuration file
    with open(arguments.config_path, 'r') as ymlfile:
        cf = yaml.load(ymlfile)

    # Set paths
    cf['Paths']['save'] = 'exp/' + arguments.exp_name
    cf['Paths']['model'] = os.path.join(cf['Paths']['save'], 'model/')
    cf['Paths']['histories'] = os.path.join(cf['Paths']['save'], 'histories/')
    #cf['Paths']['config'] = arguments.config_path

    if cf['Case'] == "train":
        train(cf)
    elif cf['Case'] == "test":
        test(cf)
    else:
        result(cf)

