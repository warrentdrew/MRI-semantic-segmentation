import keras
import keras.backend as K
from keras.models import Model
from keras.layers import *
from libs.models import denseBlock, transitionLayerPool, transitionLayerUp, resize_3D
import libs.custom_metrics
from libs.training import fit

"""
This class implements a revised version of the original work DCCN
Revision part: removing of params at high level features
"""
class DCCN_RE():
    '''
    __init__ function will build up the model with given hyperparams
    '''
    def __init__(self, in_shape, k, ls, theta, k_0, lbda=0, out_res=None, feed_pos=False, pos_noise_stdv=0):
        self.in_shape = in_shape
        self.k = k
        self.ls = ls
        self.theta = theta
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

        x = Conv3D(filters=k_0, kernel_size=(7, 7, 7), strides=(2, 2, 2), padding='same')(in_)
        shortcuts = []
        for l in ls:
            x = denseBlock(mode='3D', l=l, k=k, lbda=lbda)(x)
            shortcuts.append(x)
            k_0 = int(round((k_0 + k * l) * theta))
            x = transitionLayerPool(mode='3D', f=k_0, lbda=lbda)(x)

        if feed_pos:
            shape = x._keras_shape[1:4]
            pos = UpSampling3D(size=shape)(pos)
            x = Concatenate(axis=-1)([x, pos])

        for l, shortcut in reversed(list(zip(ls, shortcuts))):
            x = denseBlock(mode='3D', l=l, k=k, lbda=lbda)(x)
            k_0 = int(round((k_0 + k * l) * theta / 2))
            x = transitionLayerUp(mode='3D', f=k_0, lbda=lbda)(x)
            x = Concatenate(axis=-1)([shortcut, x])
        x = UpSampling3D()(x)

        if out_res is not None:
            resize = resize_3D(out_res=out_res)(x)
            cut_in = Cropping3D(3 * ((in_shape[1] - out_res) // 2,))(in_)
            x = Concatenate(axis=-1)([cut_in, resize])

        x = Conv3D(filters=3, kernel_size=(1, 1, 1))(x)
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
        cls = 3

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



