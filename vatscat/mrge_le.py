import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import keras
import keras.backend as K
from keras.models import Model
from keras.layers import *
from keras.models import load_model
from blocks import resize_3D
from blocks import transpose_conv3D_1x1_pr, conv1x1_relu, MRGE_LE_exp_blk_pr
import libs.custom_metrics as custom_metrics
from libs.training import fit
from math import log2

"""
This class implements a revised version of the original work DCCN
Revision: 
1. removing of params at high level features
2. change upsample into transpose conv
"""
class MRGE_LE():
    '''
    __init__ function will build up the model with given hyperparams
    '''
    def __init__(self, in_shape, rls, k_0, lbda=0, out_res=None, feed_pos=False, pos_noise_stdv=0):
        self.in_shape = in_shape
        self.rls = rls
        self.k_0 = k_0
        #self.out_res = out_res
        #self.feed_pos = feed_pos
        #self.pos_noise_stdv = pos_noise_stdv
        #self.lbda = lbda

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
            x = MRGE_LE_exp_blk_pr('3D', filters = k_0, dilation_max = l, lbda = lbda)(x)
            shortcuts.append(x)
            x = MaxPool3D()(x)
            k_0 = int(2 * k_0)

        k_0 = int(x.shape[-1])
        #add one dense conv at the bottleneck, shift the dense block for the decoder to make it symmetric
        '''
        x = Conv3D(filters= k_0,
                   kernel_size=(1,1,1),
                   strides=(1, 1, 1),
                   padding= 'same',
                   kernel_initializer= 'he_normal',
                   kernel_regularizer=regularizers.l2(lbda))(
                         Activation('relu')(
                         BatchNormalization()(x)))
        '''
        x = conv1x1_relu('3D', filters = k_0, initializer = 'he_normal', lbda = 0)(x)

        if feed_pos:
            shape = x._keras_shape[1:4]
            pos = UpSampling3D(size=shape)(pos)
            x = Concatenate(axis=-1)([x, pos])


        for l, shortcut in reversed(list(zip(self.rls, shortcuts))):  #start from transpose conv then mrge
            x = transpose_conv3D_1x1_pr(filters = k_0, kernel_size = (3,3,3))(x)
            x = Add()([shortcut, x])
            k_0 = int(k_0 // 2)
            x = MRGE_LE_exp_blk_pr('3D', filters=k_0, dilation_max=l, lbda=lbda)(x)

        x = Conv3D(filters=4,
                   kernel_size=(1, 1, 1),
                   strides=(1, 1, 1),
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(lbda))(x)

        if out_res is not None:
            resize = resize_3D(out_res=out_res)(x)
            cut_in = Cropping3D(3 * ((in_shape[1] - out_res) // 2,))(in_)
            x = Concatenate(axis=-1)([cut_in, resize])

        #x = Conv3D(filters=3, kernel_size=(1, 1, 1))(x)
        out = Activation('softmax', name='output_Y')(x)
        if feed_pos:
            self.model = Model([in_, in_pos], out)
        else:
            self.model = Model(in_, out)


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

        self.model.compile(optimizer='rmsprop',
                      loss=custom_metrics.jaccard_dist,
                      metrics=m1 + m2 + ['categorical_accuracy'] + [custom_metrics.jaccard_dist_discrete])
        print(' model compiled.')

    def train(self, patients_train, patients_val_slices, checkpointer, config):
        print(' training...')
        hist_object = fit(model=self.model,
                          patients_train=patients_train,
                          data_valid=patients_val_slices,
                          epochs=config.epochs,
                          batch_size=config.batch_size,
                          patient_buffer_capacity=config.patient_buffer_capacity,  # amount of patients on RAM
                          batches_per_shift=config.batches_per_shift,  # batches_per_train_epoch = batches_per_shift * len(patients_train),
                          # batches out of buffer before one shift-operation, see every patient in one epoch!
                          density=config.density,  # density for meshgrid of positions for validation data
                          border=config.border,  # distance in pixel between crops
                          callbacks=[checkpointer],  # callback (see keras documentation) for validation loss
                          mult_inputs=config.pos,  # if additional position at bottleneck, mult_inputs = True
                          empty_patient_buffer=config.empty)  # empty whole buffer, after training of one model (provide RAM for next model)

        return hist_object




