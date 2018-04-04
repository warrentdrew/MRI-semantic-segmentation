import keras
import keras.backend as K
from keras.models import Model
from keras.layers import *
"""
This class implements a revised version of the original work DCCN
Revision part: removing of params at high level features
"""
class DCCN_RE():
    def __init__(self, model, in_shape, k, ls, theta, k_0, out_res=None, feed_pos=False, pos_noise_stdv=0):
        self.in_shape = in_shape
        self.k = k
        self.ls = ls
        self.theta = theta
        self.k_0 = k_0
        self.out_res = out_res
        self.feed_pos = feed_pos
        self.pos_noise_stdv = pos_noise_stdv

        in_ = Input(shape=in_shape, name='input_X')

        if feed_pos:
            in_pos = Input(shape=(3,), name='input_position')
            pos = Reshape(target_shape=(1, 1, 3))(in_pos)
            if pos_noise_stdv != 0:
                pos = GaussianNoise(pos_noise_stdv)(pos)
            pos = BatchNormalization()(pos)

        x = Conv2D(filters=k_0, kernel_size=(7, 7), strides=(2, 2), padding='same')(in_)
        shortcuts = []
        for l in ls:
            x = denseBlock(mode='2D', l=l, k=k, lbda=lbda)(x)
            shortcuts.append(x)
            k_0 = int(round((k_0 + k * l) * theta))
            x = transitionLayerPool(mode='2D', f=k_0, lbda=lbda)(x)

        if feed_pos:
            shape = x._keras_shape[1:3]
            pos = UpSampling2D(size=shape)(pos)
            x = Concatenate(axis=-1)([x, pos])

        for l, shortcut in reversed(list(zip(ls, shortcuts))):
            x = denseBlock(mode='2D', l=l, k=k, lbda=lbda)(x)
            k_0 = int(round((k_0 + k * l) * theta / 2))
            x = transitionLayerUp(mode='2D', f=k_0, lbda=lbda)(x)
            x = Concatenate(axis=-1)([shortcut, x])
        x = UpSampling2D()(x)

        if out_res is not None:
            resize = resize_2D(out_res=out_res)(x)
            cut_in = Cropping2D(2 * ((in_shape[1] - out_res) // 2,))(in_)
            x = Concatenate(axis=-1)([cut_in, resize])

        x = Conv2D(filters=3, kernel_size=(1, 1))(x)
        out = Activation('softmax', name='output_Y')(x)
        if feed_pos:
            self.model = Model([in_, in_pos], out)
        else:
            self.model = Model(in_, out)

    def train(self):
        self.model.fit()