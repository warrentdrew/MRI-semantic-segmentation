import os
import sys
import argparse
import imp
import shutil
import yaml

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

# Memory allocation
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# Keras
import keras
import keras.backend as K
from keras import callbacks as cb
from keras.models import Sequential, Model
from keras.layers import *
K.set_image_data_format = 'channels_last'

# Import own scripts
import util
import preprocessing
from training import *
import custom_metrics
import models
import history


def train(cf):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cf['Training']['gpu_num']) # Choose GPU
    lhist = [] # List for histories
    patients_val = None


    # Custom Objects
    #------------------------------------------------------------------------------------
    # METRICS
    # TPR_cls = tp_cls/gt_cls => TPR = m1/m2
    # True positives
    m1 = [custom_metrics.metric_tp(c) for c in range(len(cf['Data']['lst_cl'])+1)]
    for j, f in enumerate(m1):
        f.__name__ = 'm_tp_c' + str(j)    
    # Ground truth
    m2 = [custom_metrics.metric_gt(c) for c in range(len(cf['Data']['lst_cl'])+1)]
    for k, f in enumerate(m2):
        f.__name__ = 'm_gt_c' + str(k)

    # LOSS
    try: getattr(custom_metrics, cf['Training']['loss_function'])
    except Exception: 
        custom_loss = False
        loss = cf['Training']['loss_function']   
    else: 
        custom_loss = True
        loss = getattr(custom_metrics, cf['Training']['loss_function'])
        loss.__name__ = cf['Training']['loss_function']

    if custom_loss: d_l = {loss.__name__ : cf['Training']['loss_function']}
    else: d_l = {}
    #------------------------------------------------------------------------------------


    for i in range(cf['Training']['iterations']):

        (patients_test, patients_train, patients_val, patients_val_slices) = load_objects(cf, i, patients_val)
        
        ################
        #  Load model  #
        ################   

        if cf['Paths']['pretrained_model'] is not None:
            print(' Load pretrained model')
            model = keras.models.load_model(filepath=cf['Paths']['pretrained_model'], 
                                            custom_objects=dict(d_l, 
                                                                **{'resize_layer' : getattr(models, 'resize_' + str(cf['Data']['dim']) + 'D')}, 
                                                                **{f.__name__ : f for f in m1}, 
                                                                **{f.__name__ : f for f in m2}))
            
                           
        else:
            print(' Load model')
            model_str = 'DenseNet' + str(cf['Data']['dim']) + 'D'
            model = getattr(models, model_str)(in_shape = (cf['Data']['train_crop_size'], ) * cf['Data']['dim'] + (len(cf['Data']['lst_ch']), ),		 
                                               k = cf['Model']['k'],		                                 
                                               ls = cf['Model']['ls'],   			                     
                                               theta = cf['Model']['theta'],			                         
                                               k_0 = cf['Model']['k_0'],				                         
                                               lbda = cf['Model']['lbda'],                                      
                                               out_res = cf['Model']['out_res'],                                  
                                               feed_pos = cf['Model']['feed_pos'],                        
                                               pos_noise_stdv = cf['Model']['pos_noise_stdv'])
            
        print(' Summary of the model:')
        model.summary()

        ###################
        #  Compile model  #
        ###################
            
        model.compile(optimizer = cf['Training']['optimizer'],                     # Optimizer for gradient descent
                      loss = loss,                          
                      metrics = m1 + m2)                                            # Metrics for evaluating segmentation results, m1 and m2 for TPR   
        print(' Model compiled!')


        # Save the model weights after each epoch if the validation loss decreased (see keras documentation, checkpointer)
        checkpointer = cb.ModelCheckpoint(filepath=cf['Paths']['weights'] + str(i) + '.hdf5', verbose=0, monitor='val_loss', save_best_only=True)


        ################
        #   Training   #
        ################
        print(' Training...')
        hist_object = fit(model = model,
                          patients_train = patients_train,
                          data_valid = patients_val_slices,
                          epochs = cf['Training']['num_epochs'],                 
                          batch_size = cf['Training']['batch_size'],         
                          patient_buffer_capacity = cf['Training']['patient_buffer_capacity'],	    
                          batches_per_shift = cf['Training']['batches_per_shift'],		   
                          density = cf['Data']['density'],			       
                          border = cf['Data']['border'],			       
                          callbacks = [checkpointer],		
                          mult_inputs = cf['Model']['feed_pos'],			    
                          empty_patient_buffer=True)	    
        print(' Training finished!')


        ################
        #   Histories  #
        ################
        print(' Save histories')
        lhist.append(hist_object.history)


    # Save histories of all iteration steps
    history.save_histories(lhist = lhist, path = cf['Paths']['histories'])


def load_objects(cf, i, patients_val):
    """Load patient objects dynamically."""

    # Drop last validation patients from RAM (k fold cross-validation)
    if ((cf['Data']['k'] is not None) and (i > 0)):
        last_val_patients = patients_val
    else:
        last_val_patients = None

    ##############################
    #  Generate patient objects  #
    ##############################

    print(' Generate patient objects.')
    patients_test, patients_train, patients_val, patients_val_slices = util.load_correct_patients(path=cf['Paths']['data'], 
                                                                                                  patients_to_take=cf['Data']['amount_test_data'], 	
                                                                                                  forget_slices=True, 	
                                                                                                  cut=cf['Data']['train_val_split'], 		    
                                                                                                  k=cf['Data']['k'], 			        
                                                                                                  perc=cf['Data']['perc'],		        
                                                                                                  iteration=i,		    
                                                                                                  last_val_patients=last_val_patients, 
                                                                                                  verbose=True)
    print(' Generated {} patient objects for training.'.format(len(patients_train)))
    
    return (patients_test, patients_train, patients_val, patients_val_slices)


def data_preprocess(cf):
    # Seed: make experiments reproducibel
    np.random.seed(cf['Training']['seed'])
    tf.set_random_seed(cf['Training']['seed'])

    if cf['Data']['decompress']:
        print( 'Decompress some MRI.')
        patients_test, patients_train, patients_val_slices, patients_val_slices = util.load_correct_patients(path=cf['Paths']['raw_data'], 
                                                                                                             patients_to_take=0, 
                                                                                                             cut=0, 
                                                                                                             verbose=True)
        # Go through all patients and decompress if neccessary       
        for patient in patients_train:
            patient.save_decomp_slices(path_to_gdcm=cf['Paths']['gdcm'], verbose=True)
        print( 'MR images are decompressed.')

    if cf['Data']['check']:
        print(' Check MRI data on errors.')
        util.check_patients(path=cf['Paths']['raw_data'], 
                            lst_ch=cf['Data']['lst_ch'], 
                            lst_cl=cf['Data']['lst_cl'],
                            path_to_gdcm=cf['Paths']['gdcm'], 
                            path_save=cf['Paths']['data'], 
                            verbose=True)
        print(' Paths to usable MR images are saved.')
        
    if not os.path.exists(cf['Paths']['save']):
        os.makedirs(cf['Paths']['save'])
    else:
        stop = input('\033[93m The folder {} already exists. Do you want to overwrite it ? ([y]/n) \033[0m'.format(cf['Paths']['save']))
        if stop == 'n':
            return

    if not os.path.exists(cf['Paths']['weights']):
        os.makedirs(cf['Paths']['weights'])
    else:
        stop = input('\033[93m The folder {} already exists. Do you want to overwrite it ? ([y]/n) \033[0m'.format(cf['Paths']['weights']))
        if stop == 'n':
            return

    if not os.path.exists(cf['Paths']['histories']):
        os.makedirs(cf['Paths']['histories'])
    else:
        stop = input('\033[93m The folder {} already exists. Do you want to overwrite it ? ([y]/n) \033[0m'.format(cf['Paths']['histories']))
        if stop == 'n':
            return

    print('-' * 75)
    print(' Config\n')
    print(' Local saving directory : ' + cf['Paths']['save'])
    print(' Model path : ' + cf['Paths']['model'])

    # Copy train script and configuration file (make experiment reproducible)
    shutil.copy(os.path.basename(sys.argv[0]), os.path.join(cf['Paths']['save'], 'train.py'))
    shutil.copy(cf['Paths']['config'], os.path.join(cf['Paths']['save'], 'config.yml'))

    # Training
    train(cf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DenseNet training')
    
    parser.add_argument('-c', '--config_path',
                        type=str,
                        default='config.yml',
                        help='Configuration file')

    parser.add_argument('-e', '--exp_name',
                        type=str,
                        default=None,
                        help='Name of experiment')

    parser.add_argument('-d', '--data_path',
                        type=str,
                        default='/media/sarah/Extern/KORA_Kopie/',
                        help='Path to data')

    arguments = parser.parse_args() 

    assert arguments.config_path is not None, 'Please provide a configuration path using' \
                                              ' -c pathname in the command line.'
    assert arguments.exp_name is not None, 'Please provide a name for the experiment using' \
                                           ' -e name in the command line'
    assert arguments.data_path is not None, 'Please provide a path to the patient data' \
                                            ' -d pathname in the command line'
   
    # Parse the configuration file
    with open(arguments.config_path, 'r') as ymlfile:
        cf = yaml.load(ymlfile)

    # Set paths
    cf['Paths']['save'] = 'exp/' + arguments.exp_name
    cf['Paths']['weights'] = os.path.join(cf['Paths']['save'], 'weights/')
    cf['Paths']['histories'] = os.path.join(cf['Paths']['save'], 'histories/')
    cf['Paths']['config'] = arguments.config_path
    cf['Paths']['data'] = arguments.data_path
    
    # You can easily launch different experiments by slightly changing cf
    data_preprocess(cf)

