import os
import shutil
import argparse
import yaml
import csv

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

# Plots
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Import own scripts
import util.util as util
import network.models as models
import network.custom_metrics as custom_metrics
import postprocess.history as history
import postprocess.metrics as metrics

def evaluation(cf, cf_exp):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cf['gpu_num']) # Choose GPU


    if not os.path.exists(cf['Paths']['exp']):
        os.makedirs(cf['Paths']['exp'])
    else:
        if not cf['Background_process']:
            stop = input('\033[93m The folder {} already exists. Do you want to overwrite it ? ([y]/n) \033[0m'.format(cf['Paths']['exp']))
            if stop == 'n':
                return

    if not os.path.exists(cf['Paths']['evaluation']):
        os.makedirs(cf['Paths']['evaluation'])
    else:
        if not cf['Background_process']:
            stop = input('\033[93m The folder {} already exists. Do you want to overwrite it ? ([y]/n) \033[0m'.format(cf['Paths']['evaluation']))
            if stop == 'n':
                return

    # Copy configuration file (make experiment reproducible)
    shutil.copy(cf['Paths']['config'], os.path.join(cf['Paths']['exp'], 'config_eval.yml'))


    ##############################
    #  Generate patient objects  #
    ##############################

    print(' Generate patient objects.')
    patients_test, patients_train, patients_val, patients_val_slices = util.load_correct_patients(path=cf_exp['Paths']['pikels'], 
                                                                                                  patients_to_take=cf_exp['Data']['amount_test_data'],
                                                                                                  cut=0)
    print(' Generated {} patient objects of test data for evaluation.'.format(len(patients_test)))


    
    #################
    #   Load Model  #
    #################
    
    # Custom Objects
    #------------------------------------------------------------------------------------
    # METRICS
    # TPR_cls = tp_cls/gt_cls => TPR = m1/m2
    # True positives
    m1 = [custom_metrics.metric_tp(c) for c in range(len(cf_exp['Data']['lst_cl'])+1)]
    for j, f in enumerate(m1):
        f.__name__ = 'm_tp_c' + str(j)    
    # Ground truth
    m2 = [custom_metrics.metric_gt(c) for c in range(len(cf_exp['Data']['lst_cl'])+1)]
    for k, f in enumerate(m2):
        f.__name__ = 'm_gt_c' + str(k)

    # PRETRAINED LOSS
    if cf['Pretrained_Model']['path'] is not None:
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
    else:
        try: getattr(custom_metrics, cf_exp['Training']['train_loss'])
        except Exception:
            custom_loss = False
            loss = cf_exp['Training']['train_loss'] # Keras loss
        else:
            custom_loss = True
            loss = getattr(custom_metrics, cf_exp['Training']['train_loss'])
            loss.__name__ = cf_exp['Training']['train_loss']
        if custom_loss: d_l = {loss.__name__ : loss}
        else: d_l = {}
        
    # RESIZE LAYER
    if cf_exp['Data']['train_crop_size'] != cf_exp['Model']['out_res']:
        res_name = 'resize_' + str(cf_exp['Data']['dim']) + 'D'
        d_res = {res_name : getattr(models, res_name)}
    else: d_res = {}
    #------------------------------------------------------------------------------------
 
    if cf['Pretrained_Model']['path'] is not None:
        print(' Load pretrained model')
        model = keras.models.load_model(filepath=cf['Pretrained_Model']['path'], 
                                        custom_objects=dict(d_pre_l, 
                                                            **d_res, 
                                                            **{f.__name__ : f for f in m1}, 
                                                            **{f.__name__ : f for f in m2}))                
    else:
        print(' Load model of the experiment.')
        model = keras.models.load_model(filepath=os.path.join(cf_exp['Paths']['model'], os.listdir(cf_exp['Paths']['model'])[cf['Iteration']]),
                                        compile=False, 
                                        custom_objects=dict(d_l, 
                                                            **d_res, 
                                                            **{f.__name__ : f for f in m1}, 
                                                            **{f.__name__ : f for f in m2}))


    ################
    #   Histories  #
    ################
    
    hist = False
    
    for key, value in cf['Histories'].items():
        if value: hist = True

    if hist:
        # load history data
        lhist = history.load_histories(path=os.path.join(cf_exp['Paths']['histories'], os.listdir(cf_exp['Paths']['histories'])[0]))

        if cf['Histories']['loss']: 
            h_loss = history.plot_loss_curve(lhist)
            h_loss.savefig(os.path.join(cf['Paths']['evaluation'], 'loss.' + cf['Savefig']['format']), 
                           format=cf['Savefig']['format'],
                           transparent=cf['Savefig']['transparent'],
                           bbox_inches=cf['Savefig']['bbox'])        
            print(' History of loss saved.')

        if cf['Histories']['train_recall']:   
            h_train_recall = history.plot_train_recall_curve(lhist, m1, m2)
            h_train_recall.savefig(os.path.join(cf['Paths']['evaluation'], 'train_recall.' + cf['Savefig']['format']), 
                                   format=cf['Savefig']['format'],
                                   transparent=cf['Savefig']['transparent'],
                                   bbox_inches=cf['Savefig']['bbox'])  
            print(' History of training recall saved.')

        if cf['Histories']['val_recall']:
            h_val_recall = history.plot_val_recall_curve(lhist, m1, m2)
            h_val_recall.savefig(os.path.join(cf['Paths']['evaluation'], 'val_recall.' + cf['Savefig']['format']), 
                                format=cf['Savefig']['format'],
                                transparent=cf['Savefig']['transparent'],
                                bbox_inches=cf['Savefig']['bbox'])
            print(' History of validation recall saved.')



    ##################################
    #  Prediction/Segmentation Plots #
    ##################################
    
    if cf['SegPlots']['plot']:
        for num in cf['SegPlots']['patient_nums']:
    
            # Ground Truth (labels) on MRI (coronal, sagittal and transverse plane)
            if cf['SegPlots']['ground_truth']:
                ground_truth = patients_test[num].plot_patient_slices(ch=cf['SegPlots']['ch'], 
                                                                      dim=cf['SegPlots']['dim'], 
                                                                      alpha=cf['SegPlots']['alpha'])
                ground_truth.savefig(os.path.join(cf['Paths']['evaluation'], str(num) + '_ground_truth' + '.' + cf['Savefig']['format']), 
                                     format=cf['Savefig']['format'],
                                     transparent=cf['Savefig']['transparent'],
                                     bbox_inches=cf['Savefig']['bbox'])  
                print(' Ground Truth plot saved.')

            # Prediction/Segmentation (of the model) on MRI 
            if cf['SegPlots']['prediction_on_mri']:
                prediction = patients_test[num].plot_prediction_on_patient(model=model, 
                                                                           batch_size=cf['Prediction']['batch_size'], 
                                                                           ch=cf['SegPlots']['ch'],
                                                                           dim=cf['SegPlots']['dim'], 
                                                                           alpha=cf['SegPlots']['alpha'])
                prediction.savefig(os.path.join(cf['Paths']['evaluation'], str(num) + '_prediction' + '.' + cf['Savefig']['format']), 
                                   format=cf['Savefig']['format'],
                                   transparent=cf['Savefig']['transparent'],
                                   bbox_inches=cf['Savefig']['bbox'])  
                print(' Segmentation plot saved.')

            # Prediction/Segmentation (left) vs Ground-Truth (right)
            if cf['SegPlots']['prediction_vs_gt']:
                prediction_vs_ground = patients_test[num].plot_prediction_vs_ground_truth(model=model,
                                                                                          batch_size=cf['Prediction']['batch_size'],
                                                                                          depth=cf['SegPlots']['dim'][2])
                prediction_vs_ground.savefig(os.path.join(cf['Paths']['evaluation'], str(num) + '_prediction_vs_truth' + '.' + cf['Savefig']['format']), 
                                             format=cf['Savefig']['format'],
                                             transparent=cf['Savefig']['transparent'],
                                             bbox_inches=cf['Savefig']['bbox'])  
                print(' Segmentation vs. Ground Truth plot saved.')

            # Heatmap (to show how reliable/robust the prediction/segmentation is (prediction for each voxel is in range(0,1)))
            if cf['SegPlots']['heatmap']:
                heatmap = patients_test[num].heatmap(model=model, 
                                                     batch_size=cf['Prediction']['batch_size'],
                                                     depth=cf['SegPlots']['dim'][2], 
                                                     cls=cf['SegPlots']['heatmap_cls'])
                heatmap.savefig(os.path.join(cf['Paths']['evaluation'], str(num) + '_heatmap' + '.' + cf['Savefig']['format']), 
                                format=cf['Savefig']['format'],
                                transparent=cf['Savefig']['transparent'],
                                bbox_inches=cf['Savefig']['bbox'])  
                print(' Heatmap plot saved.') 

            if cf['SegPlots']['ground_truth_coronal']:
                path = cf['Paths']['evaluation'] + str(num) + '_ground_truth_coronal/'
                os.makedirs(path)
                dicoms, _ = patients_test[num].load_slices()
                dim_z = dicoms.shape[-2]
                for i in range(dim_z):
                    cor_gt = patients_test[num].plot_patient_coronal(ch='water', dim_z=i, alpha=0.3)
                    print(' Save ground truth coronal fig in z ' + str(i))
                    cor_gt.savefig(path + str(i) + '.' + cf['Savefig']['format'], 
                                   format=cf['Savefig']['format'],
                                   transparent=cf['Savefig']['transparent'],
                                   bbox_inches=cf['Savefig']['bbox'])
                    plt.close('all')
                print(' Coronal ground truths saved.')  

            if cf['SegPlots']['prediction_coronal']:
                path = cf['Paths']['evaluation'] + str(num) + '_prediction_coronal/'
                print(path)
                os.makedirs(path)
                dicoms, _ = patients_test[num].load_slices()
                dim_z = dicoms.shape[-2]
                for i in range(dim_z):
                    cor_pred = patients_test[num].plot_prediction_on_patient_coronal(model=model,
                                                                                     batch_size=cf['Prediction']['batch_size'],
                                                                                     ch='water', 
                                                                                     dim_z=i,
                                                                                     alpha=0.3)
                    print(' Save prediction coronal fig at z ' + str(i))
                    cor_pred.savefig(path + str(i) + '.' + cf['Savefig']['format'], 
                                     format=cf['Savefig']['format'],
                                     transparent=cf['Savefig']['transparent'],
                                     bbox_inches=cf['Savefig']['bbox'])
                    plt.close('all')
                print(' Coronal predictions saved.')  
                                        


    ################
    #   Metrics   #
    ################
    
    conf_ma = False
    
    for key, value in cf['Metrics'].items():
        if value: conf_ma = True

    if conf_ma:
    
        print(' Compute confusion matrix of {} test patients ...'.format(cf['Metrics']['patient_num_end'] - cf['Metrics']['patient_num_start']))
        matrix, accuracy, true_positives, true_negatives, false_positives, false_negatives = metrics.confusion_matrix(patients=patients_test[cf['Metrics']['patient_num_start']:cf['Metrics']['patient_num_end']], 
                                                                                                                      model=model,
                                                                                                                      batch_size=cf['Prediction']['batch_size'], 
                                                                                                                      max_crop_size=32, 
                                                                                                                      discrete=True, 
                                                                                                                      normalized=True)
        # Generate csv table with metrics
        with open(os.path.join(cf['Paths']['evaluation'], 'metrics.csv'), 'w') as csvfile:
            # Same sequence of classes as generated in util.py
            cf_exp['Data']['lst_cl'].sort()
            fieldnames = ['metric'] + ['background'] + ['class ' + c for c in cf_exp['Data']['lst_cl']] + ['overall']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

        # Derivations from a confusion matrix
        if cf['Metrics']['sensitivity']:
            sensitivity = metrics.sensitivity(true_positives, false_negatives)
            with open(os.path.join(cf['Paths']['evaluation'], 'metrics.csv'), 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(dict({'metric': 'sensitivity', 'background': sensitivity[0]},
                                     **{'class ' + c : sensitivity[i+1]  for i,c in enumerate(cf_exp['Data']['lst_cl'])}))

        if cf['Metrics']['specificity']:
            specificity = metrics.specificity(true_negatives, false_positives)
            with open(os.path.join(cf['Paths']['evaluation'], 'metrics.csv'), 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames)
                writer.writerow(dict({'metric': 'specificity', 'background': specificity[0]},
                                     **{'class ' + c : specificity[i+1]  for i,c in enumerate(cf_exp['Data']['lst_cl'])}))

        if cf['Metrics']['precision']:
            precision = metrics.precision(true_positives, false_positives)
            with open(os.path.join(cf['Paths']['evaluation'], 'metrics.csv'), 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames)
                writer.writerow(dict({'metric': 'precision', 'background': precision[0]},
                                     **{'class ' + c : precision[i+1]  for i,c in enumerate(cf_exp['Data']['lst_cl'])}))
   
        if cf['Metrics']['false_negative_rate']:
            false_negative_rate = metrics.false_negative_rate(true_positives, false_negatives)
            with open(os.path.join(cf['Paths']['evaluation'], 'metrics.csv'), 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames)
                writer.writerow(dict({'metric': 'false_negative_rate', 'background': false_negative_rate[0]},
                                     **{'class ' + c : false_negative_rate[i+1]  for i,c in enumerate(cf_exp['Data']['lst_cl'])}))

        if cf['Metrics']['false_positive_rate']:
            false_postive_rate = metrics.false_positive_rate(true_negatives, false_positives)
            with open(os.path.join(cf['Paths']['evaluation'], 'metrics.csv'), 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames)
                writer.writerow(dict({'metric': 'false_postive_rate', 'background': false_postive_rate[0]},
                                     **{'class ' + c : false_postive_rate[i+1]  for i,c in enumerate(cf_exp['Data']['lst_cl'])}))
   
        if cf['Metrics']['dice']:
            dice = metrics.dice(true_positives, false_positives, false_negatives)
            with open(os.path.join(cf['Paths']['evaluation'], 'metrics.csv'), 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames)
                writer.writerow(dict({'metric': 'dice', 'background': dice[0]},
                                     **{'class ' + c : dice[i+1]  for i,c in enumerate(cf_exp['Data']['lst_cl'])}))

        if cf['Metrics']['jaccard']:
            jaccard = metrics.jaccard(true_positives, false_positives, false_negatives)
            with open(os.path.join(cf['Paths']['evaluation'], 'metrics.csv'), 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames)
                writer.writerow(dict({'metric': 'jaccard', 'background': jaccard[0]},
                                     **{'class ' + c : jaccard[i+1]  for i,c in enumerate(cf_exp['Data']['lst_cl'])}))
        if cf['Metrics']['accuracy']:
            with open(os.path.join(cf['Paths']['evaluation'], 'metrics.csv'), 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames) 
                writer.writerow(dict({'metric': 'accuracy', 'overall' : accuracy}))

        print(' All metrics in {} saved.'.format(os.path.join(cf['Paths']['evaluation'], 'metrics.csv')))

        np.save(file=os.path.join(cf['Paths']['evaluation'], 'conf_matrix.npy'), arr=matrix)
        print(' Confusion matrix as .npy saved in {}.'.format(os.path.join(cf['Paths']['evaluation'], 'metrics.csv')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DenseNet evaluation')
    
    parser.add_argument('-c', '--config_path',
                        type=str,
                        default='config/config_eval.yml',
                        help='Configuratiyon file')

    parser.add_argument('-e', '--exp_name',
                        type=str,
                        default=None,
                        help='Name of experiment')

    parser.add_argument('-i', '--iter_exp',
                        type=int,
                        default=0,
                        help='Which iteration step of the experiment you want to evaluate?')

    parser.add_argument('-eval', '--eval_folder',
                        type=str,
                        default='evaluation/',
                        help='Name of the folder to save evaluation.')

    arguments = parser.parse_args() 

    assert arguments.config_path is not None, 'Please provide a configuration path using' \
                                              ' -c pathname in the command line.'
    assert arguments.exp_name is not None, 'Please provide a name of the experiment you want to evaluate' \
                                           ' -e name in the command line'
    assert arguments.eval_folder is not None, 'Please provide a name for the evaluation folder' \
                                           ' -eval name in the command line'
    assert arguments.iter_exp is not None, 'Please provide an iteration step of the experiment using' \
                                           ' -i name in the command line'
   
    
    # Parse the configuration file
    with open(arguments.config_path, 'r') as ymlfile:
        cf = yaml.load(ymlfile)

    cf['Iteration'] = arguments.iter_exp

    # Set paths
    cf['Paths']['exp'] = 'exp/' + arguments.exp_name
    cf['Paths']['evaluation'] = os.path.join(cf['Paths']['exp'], arguments.eval_folder)
    cf['Paths']['config'] = arguments.config_path

    # Parse configuration file of experiment
    if cf['Pretrained_Model']['path'] is not None:
        cf_exp = {'Paths' : {'pikels' : cf['Pretrained_Model']['pikels']}, 
                  'Data' : {'amount_test_data' : cf['Pretrained_Model']['amount_test_data'], 
                            'lst_cl' : cf['Pretrained_Model']['lst_cl'],
                            'train_crop_size': cf['Pretrained_Model']['train_crop_size'],
                            'dim' : cf['Pretrained_Model']['dim']},
                  'Model' : {'out_res' : cf['Pretrained_Model']['out_res']}}
    else:
        with open(os.path.join(cf['Paths']['exp'], 'config.yml'), 'r') as ymlfile_exp:
            cf_exp = yaml.load(ymlfile_exp)
    
    # You can easily launch different evaluations of experiments by slightly changing cf
    evaluation(cf, cf_exp)
   
