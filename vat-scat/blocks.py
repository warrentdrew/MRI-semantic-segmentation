import keras
import keras.backend as K
from keras.models import Model
from keras.layers import *
from libs.model import denseConv

'''
Dilated conv idea and aspp from deeplab paper
'''
def DilatedConv(mode, filters, kernel_size,  strides, dilation_rate, initializer, lbda, padding='same'):
    #perform a dilated conv on feature map
    #Args:
    #   mode: 2D or 3D
    #   in_shape: input shape
    #   filter: num of output feature map
    #   dilation_rate: the distance between two pixels covered by the kernel
    #   lbda: param for weigth decay
    if mode == '2D':
        return lambda x : Conv2D(filters,
                                 kernel_size,
                                 strides = strides,
                                 padding = padding,
                                 dilation_rate = dilation_rate,
                                 kernel_initializer = initializer,
                                 kernel_regularizer = regularizers.l2(lbda))(
                          Activation('relu')(
                          BatchNormalization()(x))) #remove bias regularizer, usually only weights needs to be regularized
    else:
        return lambda x : Conv3D(filters,
                                 kernel_size,
                                 strides=strides,
                                 padding=padding,
                                 dilation_rate=dilation_rate,
                                 kernel_initializer = initializer,
                                 kernel_regularizer=regularizers.l2(lbda))(
                          Activation('relu')(
                          BatchNormalization()(x))) #remove bias regularizer, usually only weights needs to be regularized

def ASPP(mode, filters, strides, initializer, dilation_rate_list, image_level_pool_size, lbda,  padding = 'same'):
    # ASPP: atrous spatial pyramid pooling
    # the version proposed in deeplab v3, with
    # (a) pyramid part: 1 1x1 conv and 3 paralleled 3x3 dilated conv with BN
    # (b) image level: global pooling for the input
    # in each branch, the feature map first go through conv, then BN, then Relu
    # Args:
    #   image_level_pool_size :  the size of the feature used after pooling
    if mode == '2D':
        def ASPP_instance(x):
            ## pyramid part
            pyramid_1x1 = Activation('relu')(BatchNormalization()(Conv2D(filters,
                                                                         kernel_size = (1,1),
                                                                         strides = strides,
                                                                         padding = padding,
                                                                         kernel_initializer=initializer,
                                                                         kernel_regularizer=regularizers.l2(lbda))(x)))

            branch = [pyramid_1x1]
            for rate in dilation_rate_list:
                branch.append(Activation('relu')(BatchNormalization()(DilatedConv(mode, filters, (3,3), rate, lbda)(x))))
            ##image level part
            image_level_feature = Conv2D(filters,
                                        kernel_size=(1, 1),
                                        strides=strides,
                                        padding=padding,
                                        kernel_regularizer=regularizers.l2(lbda))(
                                            AveragePooling2D(pool_size=image_level_pool_size, padding='valid')(x))
            image_level_feature = K.tf.image.resize_bilinear(images=image_level_feature, size=image_level_pool_size, align_corners=True)
            branch.append(image_level_feature)
            branch_logit = Concatenate(axis = -1)(branch)
            #add an 1x1 conv for final output
            output = Conv2D(filters,
                           kernel_size=(1, 1),
                           strides=strides,
                           padding=padding,
                           kernel_regularizer=regularizers.l2(lbda))(branch_logit)
            return output

        return ASPP_instance
    else:
        def ASPP_instance(x):
            ## pyramid part
            pyramid_1x1 = Activation('relu')(BatchNormalization()(Conv3D(filters,
                                                                         kernel_size = (1,1,1),
                                                                         strides = strides,
                                                                         padding = padding,
                                                                         kernel_regularizer = regularizers.l2(lbda))(x)))

            branch = [pyramid_1x1]
            for rate in dilation_rate_list:
                branch.append(Activation('relu')(BatchNormalization()(DilatedConv(mode, filters, (3,3,3), rate, lbda)(x))))
            ##image level part
            image_level_feature = Conv3D(filters,
                                        kernel_size=(1,1,1),
                                        strides=strides,
                                        padding=padding,
                                        kernel_regularizer=regularizers.l2(lbda))(
                                            AveragePooling3D(pool_size=image_level_pool_size, padding='valid')(x))
            image_level_feature = K.tf.image.resize_bilinear(images=image_level_feature, size=image_level_pool_size, align_corners=True)
            branch.append(image_level_feature)
            branch_logit = Concatenate(axis = -1)(branch)
            #add an 1x1 conv for final output
            output = Conv3D(filters,
                           kernel_size=(1,1,1),
                           strides=strides,
                           padding=padding,
                           kernel_regularizer=regularizers.l2(lbda))(branch_logit)
            return output

        return ASPP_instance
'''
Dilated conv idea to feature extracting in low level
merge-and-run, create a local path and a global path
'''
def MR_local_path(mode, filters, kernel_size,  strides, initializer, lbda, padding='same'):
    # implement a normal residual path in a residual block, which is used as a path in the merge and run net
    # the path is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    # bn -> relu -> conv
    if mode == '2D':
        return lambda x : Conv2D(filters,
                                 kernel_size= kernel_size,
                                 strides= strides,
                                 padding= padding,
                                 kernel_initializer= initializer,
                                 kernel_regularizer = regularizers.l2(lbda))(
                                    Activation('relu')(BatchNormalization()(
                                    Conv2D(filters,
                                           kernel_size,
                                           strides,
                                           padding,
                                           kernel_initializer= initializer,
                                           kernel_regularizer= regularizers.l2(lbda))(Activation('relu')(BatchNormalization()(x))))))
    else:
        return lambda x: Conv3D(filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding=padding,
                                kernel_initializer=initializer,
                                kernel_regularizer=regularizers.l2(lbda))(
                                    Activation('relu')(BatchNormalization()(
                                        Conv3D(filters,
                                               kernel_size,
                                               strides,
                                               padding,
                                               kernel_initializer=initializer,
                                               kernel_regularizer=regularizers.l2(lbda))(Activation('relu')(BatchNormalization()(x))))))

def MR_global_path(mode, filters, kernel_size,  strides, dilation_rate, initializer, lbda, padding='same'):
    # a novel idea, to include a global path in the merge and run net implemented with dilated conv
    return lambda x : DilatedConv(mode, filters, kernel_size, strides, dilation_rate, initializer, lbda, padding)(
                            Activation('relu')(BatchNormalization()(x)))


def MR_block(mode, pos, filters, kernel_size, strides, initializer, lbda, padding='same'):
    # implementation of the merge-and-run block in https://arxiv.org/pdf/1611.07718.pdf
    def MR_instance(x,y):
        mid = add([x,y])
        x_conv = MR_local_path(mode, filters, kernel_size,  strides, initializer, lbda, padding)(x)
        y_conv = MR_local_path(mode, filters, kernel_size,  strides, initializer, lbda, padding)(y)
        if pos == 'end':
            out = add([add([x_conv,y_conv]),mid])
            return out
        else:
            x_out = add([x_conv,mid])
            y_out = add([y_conv,mid])
            return x_out, y_out

    return MR_instance


def MR_GE_block(mode, pos, filters, kernel_size,  strides, dilation_rate, initializer, lbda, padding='same'):
    # GE stands for global enhanced
    # a novel idea for combining local path with global path
    def MR_instance(x,y):
        mid = add([x,y])
        x_conv = MR_local_path(mode, filters, kernel_size, strides, initializer, lbda, padding)(x)
        y_conv = MR_global_path(mode, filters, kernel_size, strides, dilation_rate, initializer, lbda, padding)(y)
        if pos == 'end':
            out = add([add([x_conv,y_conv]),mid])
            return out
        else:
            x_out = add([x_conv,mid])
            y_out = add([y_conv,mid])
            return x_out, y_out
    return MR_instance


"""
Revised blocks for dccn_re
"""

def transitionLayerTransposeUp(mode, f, lbda):
    if mode == '2D':
        return lambda x: Conv2DTranspose(filters=f, kernel_size=(3, 3), strides=(2, 2),
                                         padding="same", kernel_regularizer = regularizers.l2(lbda))(
                         denseConv('2D', f, 1, lbda)(x))
    else:
        return lambda x: Conv2DTranspose(filters=f, kernel_size=(3, 3, 3), strides=(2, 2, 2),
                                         padding="same", kernel_regularizer = regularizers.l2(lbda))(
                         denseConv('3D', f, 1, lbda)(x))



