import os
import sys
import argparse
import shutil
import yaml

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

# Memory allocation
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#set_session(tf.Session(config=config))

# Keras
import keras
import keras.backend as K
from keras import callbacks as cb
from keras.models import Sequential, Model
from keras.layers import *
K.set_image_data_format = 'channels_last'

# Import own scripts
import util.util as util
import preprocess.preprocessing as preprocessing
import network.training as training
import network.custom_metrics as custom_metrics
import network.models as models
import postprocess.history as history

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

    # PRETRAINED LOSS
    try: getattr(custom_metrics, cf['Pretrained_Model']['pretrained_loss'])
    except Exception: 
        custom_pre_loss = False
        pre_loss = cf['Pretrained_Model']['pretrained_loss']   
    else: 
        custom_pre_loss = True
        pre_loss = getattr(custom_metrics, cf['Pretrained_Model']['pretrained_loss'])
        pre_loss.__name__ = cf['Pretrained_Model']['pretrained_loss']

    if custom_pre_loss: d_pre_l = {pre_loss.__name__ : pre_loss}
    else: d_pre_l = {}

    # TRAIN LOSS
    try: getattr(custom_metrics, cf['Training']['train_loss'])
    except Exception: 
        loss = cf['Training']['train_loss'] # Keras loss
    else: 
        loss = getattr(custom_metrics, cf['Training']['train_loss'])

    # RESIZE LAYER
    if cf['Pretrained_Model']['out_res'] is not None:
        res_name = 'resize_' + str(cf['Pretrained_Model']['dim']) + 'D'
        d_res = {res_name : getattr(models, res_name)}
    #------------------------------------------------------------------------------------


    for i in range(cf['Training']['iterations']):

        (patients_test, patients_train, patients_val, patients_val_slices) = load_objects(cf, i, patients_val)
        
        ################
        #  Load model  #
        ################   
        
        print('-' * 75)
        print(' Model\n')

        if cf['Pretrained_Model']['path'] is not None:
            print(' Load pretrained model')
            model = keras.models.load_model(filepath=cf['Pretrained_Model']['path'], 
                                            custom_objects=dict(d_pre_l, 
                                                                **d_res, 
                                                                **{f.__name__ : f for f in m1}, 
                                                                **{f.__name__ : f for f in m2}))                
        else:
            print(' Load model')
            model_str = cf['Model']['name'] + str(cf['Data']['dim']) + 'D'
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
                      metrics = m1 + m2)                                           # Metrics for evaluating segmentation results, m1 and m2 for TPR   
        print(' Model compiled!')


        # Save the model after each epoch if the validation loss decreased (see keras documentation, checkpointer)
        checkpointer = cb.ModelCheckpoint(filepath=cf['Paths']['model'] + str(i) + '.h5', verbose=0, monitor='val_loss', save_best_only=True)


        ################
        #   Training   #
        ################
        print('-' * 75)
        print(' Training...')
        hist_object = training.fit(model = model,
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
        lhist.append(hist_object.history)


    # Save histories of all iteration steps
    history.save_histories(lhist=lhist, path=os.path.join(cf['Paths']['histories'], 'h.pkl'))
    print(' Histories saved.')


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

    print(' Generate patient objects and load validation data.')
    patients_test, patients_train, patients_val, patients_val_slices = util.load_correct_patients(path=cf['Paths']['pikels'], 
                                                                                                  patients_to_take=cf['Data']['amount_test_data'], 	
                                                                                                  forget_slices=True, 	
                                                                                                  cut=cf['Data']['train_val_split'], 		    
                                                                                                  k=cf['Data']['k'], 			        
                                                                                                  perc=cf['Data']['perc'],		        
                                                                                                  iteration=i,		    
                                                                                                  last_val_patients=last_val_patients, 
                                                                                                  verbose=True)
    print('-' * 75)
    print(' Data\n')
    print(' Generated {} patient objects for training.'.format(len(patients_train)))
    print(' Loaded MRI of {} patients for validation.'.format(len(patients_val)))
    
    return (patients_test, patients_train, patients_val, patients_val_slices)


def data_preprocess(cf):
    # Seed: make experiments reproducibel
    np.random.seed(cf['Training']['seed'])
    tf.set_random_seed(cf['Training']['seed'])

    if cf['Data']['check']:
        print(' Check MRI data on errors and save directories to usable ones.')
        util.check_patients(dataset=cf['Data']['dataset'],
                            path_dicoms=cf['Paths']['dicoms'],
                            path_labels=cf['Paths']['labels'], 
                            lst_ch=cf['Data']['lst_ch'], 
                            lst_cl=cf['Data']['lst_cl'],
                            path_to_gdcm=cf['Paths']['gdcm'], 
                            path_save=cf['Paths']['dicoms'], 
                            verbose=True)
        print(' Paths to usable MR images are saved.')

    if cf['Data']['decompress']:
        print( 'Decompress some MRI.')
        patients_test, patients_train, patients_val_slices, patients_val_slices = util.load_correct_patients(path=cf['Paths']['dicoms'], 
                                                                                                             patients_to_take=0, 
                                                                                                             cut=0, 
                                                                                                             verbose=True)
        # Go through all patients and decompress if neccessary       
        for patient in patients_train:
            patient.save_decomp_slices(path_to_gdcm=cf['Paths']['gdcm'], verbose=True)
        print( 'MR images are decompressed.')
        
    if not os.path.exists(cf['Paths']['save']):
        os.makedirs(cf['Paths']['save'])
    else:
        if not cf['Training']['background_process']:
            stop = input('\033[93m The folder {} already exists. Do you want to overwrite it ? ([y]/n) \033[0m'.format(cf['Paths']['save']))
            if stop == 'n':
                return

    if not os.path.exists(cf['Paths']['model']):
        os.makedirs(cf['Paths']['model'])
    else:
        if not cf['Training']['background_process']:
            stop = input('\033[93m The folder {} already exists. Do you want to overwrite it ? ([y]/n) \033[0m'.format(cf['Paths']['model']))
            if stop == 'n':
                return

    if not os.path.exists(cf['Paths']['histories']):
        os.makedirs(cf['Paths']['histories'])
    else:
        if not cf['Training']['background_process']:
            stop = input('\033[93m The folder {} already exists. Do you want to overwrite it ? ([y]/n) \033[0m'.format(cf['Paths']['histories']))
            if stop == 'n':
                return

    print('-' * 75)
    print(' Config\n')
    print(' Local saving directory : ' + cf['Paths']['save'])

    # Copy train script and configuration file (make experiment reproducible)
    shutil.copy(os.path.basename(sys.argv[0]), os.path.join(cf['Paths']['save'], 'train.py'))
    shutil.copy(cf['Paths']['config'], os.path.join(cf['Paths']['save'], 'config.yml'))

    # Extend the configuration file with new entries
    with open(os.path.join(cf['Paths']['save'], 'config.yml'), "w") as ymlfile:
        yaml.dump(cf, ymlfile)


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
    cf['Paths']['config'] = arguments.config_path
    
    # You can easily launch different experiments by slightly changing cf
    data_preprocess(cf)

    # Training
    train(cf)
