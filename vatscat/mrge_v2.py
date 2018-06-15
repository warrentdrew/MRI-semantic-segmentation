import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import keras
import keras.backend as K
from keras.models import Model
from keras.layers import *
from keras.models import load_model
from blocks import denseBlock, transitionLayerPool, resize_3D
from blocks import transitionLayerTransposeUp
from blocks import MRGE_exp_block
import libs.custom_metrics as custom_metrics
from libs.training import fit


"""
This class implements a revised version of the original work DCCN
Revision: 
1. removing of params at high level features
2. change upsample into transpose conv
"""
class MRGE_V2():
    '''
    __init__ function will build up the model with given hyperparams
    '''
    def __init__(self, in_shape, rls, k_0, pooling_num, lbda=0, out_res=None):
        self.in_shape = in_shape
        self.rls = rls
        self.k_0 = k_0
        self.pooling_num = pooling_num


        in_ = Input(shape=in_shape, name='input_X')

        shortcuts = []
        x = in_

        for l in rls[:pooling_num]:            #rls for mege_v2 is [16,8,4,4,4]    params = (mode, filters, dilation_rate, lbda, initializer, padding='same')
            x = MRGE_exp_block('3D', filters = k_0, dilation_max = l, lbda = lbda)(x)
            shortcuts.append(x)
            x = MaxPool3D()(x)
            k_0 = int(2 * k_0)

        for l in rls[pooling_num:-1]:
            x = MRGE_exp_block('3D', filters=k_0, dilation_max=l, lbda=lbda)(x)
            shortcuts.append(x)
            k_0 = int(2 * k_0)


        x = MRGE_exp_block('3D', filters=k_0, dilation_max=rls[-1], lbda=lbda)(x)
        k_0 = int(x.shape[-1]) // 2
        print('k_0:', k_0)

        reverse_rls = list(reversed(self.rls))
        reverse_short = list(reversed(shortcuts))

        for i, l in enumerate(reverse_rls[:(-pooling_num-1)]):
            x = MRGE_exp_block('3D', filters=k_0, dilation_max=l, lbda=lbda)(x)
            x = Add()([reverse_short[i], x])
            k_0 = int(k_0 // 2)


        x = MRGE_exp_block('3D', filters=k_0, dilation_max=reverse_rls[(-pooling_num-1)], lbda=lbda)(x)

        print('k_0:', k_0)
        for i, l in enumerate(reverse_rls[-pooling_num:]):
            x = Conv3DTranspose(filters=k_0, kernel_size=(3, 3, 3), strides=(2, 2, 2),
                                padding="same", kernel_initializer='he_normal',
                                kernel_regularizer=regularizers.l2(lbda))(x)
            print('shape:', reverse_short[-pooling_num + i].shape[-1])
            x = Add()([reverse_short[-pooling_num + i], x])
            x = MRGE_exp_block('3D', filters=k_0, dilation_max=l, lbda=lbda)(x)
            k_0 = int(k_0 // 2)


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




