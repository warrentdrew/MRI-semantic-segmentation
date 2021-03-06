import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import keras
import keras.backend as K
from keras.models import Model
from keras.layers import *
from keras.models import load_model
from keras.utils import multi_gpu_model
from libs.blocks import resize_3D
from libs.blocks import MR_GE_block, MR_block_split, MR_GE_block_merge
import network.custom_metrics as custom_metrics
from network.training import fit
from math import log2
from network.training import fit_resume
import tensorflow as tf

""" 
This class implements a revised version of the original work DCCN
Revision: 
1. removing of params at high level features
2. change upsample into transpose conv
"""
class MRGE():
    '''
    __init__ function will build up the model with given hyperparams
    '''
    def __init__(self, in_shape, rls, k_0, multi = False, lbda=0, out_res=None, feed_pos=False, pos_noise_stdv=0):
        self.in_shape = in_shape
        self.rls = rls
        self.k_0 = k_0

        in_ = Input(shape=in_shape, name='input_X')

        if feed_pos:
            in_pos = Input(shape=(3,), name='input_position')
            pos = Reshape(target_shape=(1, 1, 1, 3))(in_pos)
            if pos_noise_stdv != 0:
                pos = GaussianNoise(pos_noise_stdv)(pos)
            pos = BatchNormalization()(pos)

        shortcuts = []
        x = in_

        for l in rls:            #rls for mege_net is [8,4,2,1,1]    params = (mode, filters, dilation_rate, lbda, initializer, padding='same')
            x, y = MR_block_split(k_0, lbda)(x)
            block_num = int(log2(l) + 1)
            rate_list = [2 ** i for i in range(block_num)]
            for rate in rate_list[:-1]:
                x, y = MR_GE_block('3D', filters= k_0, dilation_rate = rate, lbda = lbda)(x,y)
            x = MR_GE_block_merge('3D', filters = k_0, dilation_rate = rate_list[-1], lbda=lbda)(x,y)
            shortcuts.append(x)
            x = MaxPool3D()(x)
            k_0 = int(2 * k_0)

        k_0 = int(x.shape[-1])
        #add one dense conv at the bottleneck, shift the dense block for the decoder to make it symmetric
        x = Conv3D(filters= k_0,
                   kernel_size=(1,1,1),
                   strides=(1, 1, 1),
                   padding= 'same',
                   kernel_initializer= 'he_normal',
                   kernel_regularizer=regularizers.l2(lbda))(
                         Activation('relu')(
                         BatchNormalization()(x)))

        if feed_pos:
            shape = x._keras_shape[1:4]
            pos = UpSampling3D(size=shape)(pos)
            x = Concatenate(axis=-1)([x, pos])

        for l, shortcut in reversed(list(zip(self.rls, shortcuts))):  #start from transpose conv then mrge
            x = Conv3DTranspose(filters=k_0, kernel_size=(3, 3, 3), strides=(2, 2, 2),
                            padding="same", kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(lbda))(x)
            x = Add()([shortcut, x])
            k_0 = int(k_0 // 2)
            x, y = MR_block_split(k_0, lbda)(x)
            block_num = int(log2(l) + 1)
            rate_list = [2 ** i for i in range(block_num)]
            for rate in rate_list[:-1]:
                x, y = MR_GE_block('3D', filters=k_0, dilation_rate=rate, lbda=lbda)(x, y)
            x = MR_GE_block_merge('3D', filters=k_0, dilation_rate=rate_list[-1], lbda=lbda)(x, y)

        x = Conv3D(filters=4,
                   kernel_size=(1, 1, 1),
                   strides=(1, 1, 1),
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(lbda))(
                Activation('relu')(
                BatchNormalization()(x)))

        if out_res is not None:
            resize = resize_3D(out_res=out_res)(x)
            cut_in = Cropping3D(3 * ((in_shape[1] - out_res) // 2,))(in_)
            x = Concatenate(axis=-1)([cut_in, resize])

        #x = Conv3D(filters=3, kernel_size=(1, 1, 1))(x)
        out = Activation('softmax', name='output_Y')(x)
        if feed_pos:
            self.model = Model([in_, in_pos], out)
            if multi:
                self.model = multi_gpu_model(self.model, gpus = 2)
        else:
            self.model = Model(in_, out)
            if multi:
                self.model = multi_gpu_model(self.model, gpus = 2)

    '''
    compile model
    settings for true-positive-rate (TPR)
    '''
    def compile(self):
        cls = 4

        m1 = [custom_metrics.metric_tp(c) for c in range(cls)]
        for j, f in enumerate(m1):
            f.__name__ = 'm_tp_c' + str(j)

        m2 = [custom_metrics.metric_gt(c) for c in range(cls)]
        for k, f in enumerate(m2):
            f.__name__ = 'm_gt_c' + str(k)

        mrecall = [custom_metrics.metric_recall(c) for c in range(cls)]
        mrecall[0].__name__ = 'recall_bg'
        mrecall[1].__name__ = 'recall_lt'
        mrecall[2].__name__ = 'recall_vat'
        mrecall[3].__name__ = 'recall_scat'

        self.model.compile(optimizer='rmsprop',
                      loss = custom_metrics.focal_loss,
                      metrics=m1 + m2 + mrecall + ['categorical_accuracy'] + [custom_metrics.jaccard_dist, custom_metrics.jaccard_dist_discrete] + [custom_metrics.my_accuracy])
        print(' model compiled.')

    def train(self, patients_train, patients_val_slices, checkpointer, cf):
        print(' training...')
        hist_object = fit(model=self.model,
                          patients_train=patients_train,
                          data_valid=patients_val_slices,
                          epochs=cf['Training']['num_epochs'],
                          batch_size=cf['Training']['batch_size'],
                          patient_buffer_capacity=cf['Training']['patient_buffer_capacity'],  # amount of patients on RAM
                          batches_per_shift=cf['Training']['batches_per_shift'],  # batches_per_train_epoch = batches_per_shift * len(patients_train),
                          # batches out of buffer before one shift-operation, see every patient in one epoch!
                          density=cf['Training']['density'],  # density for meshgrid of positions for validation data
                          border=cf['Preprocess']['border'],  # distance in pixel between crops
                          callbacks=[checkpointer],  # callback (see keras documentation) for validation loss
                          mult_inputs=cf['Model']['feed_pos'],  # if additional position at bottleneck, mult_inputs = True
                          empty_patient_buffer=cf['Training']['empty'])  # empty whole buffer, after training of one model (provide RAM for next model)

        return hist_object






