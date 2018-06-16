import keras
import keras.backend as K
from keras.models import Model
from keras.layers import *
from math import log2
'''
basic blocks
'''
def bn_relu_conv_3x3(mode, filters, initializer, lbda, padding='same'):
    if mode == '2D':
        return lambda x : Conv2D(filters,
                          kernel_size = (3, 3),
                          strides= (1, 1),
                          padding=padding,
                          kernel_initializer = initializer,
                          kernel_regularizer=regularizers.l2(lbda))(
                              Activation('relu')(
                              BatchNormalization()(x)))
    else:
        return lambda x: Conv3D(filters,
                         kernel_size=(3, 3, 3),
                         strides=(1, 1, 1),
                         padding=padding,
                         kernel_initializer=initializer,
                         kernel_regularizer=regularizers.l2(lbda))(
                                Activation('relu')(
                                BatchNormalization()(x)))

def bn_relu_conv_1x1(mode, filters, initializer, lbda, padding='same'):
    if mode == '2D':
        return lambda x : Conv2D(filters,
                          kernel_size = (1, 1),
                          strides= (1, 1),
                          padding=padding,
                          kernel_initializer = initializer,
                          kernel_regularizer=regularizers.l2(lbda))(
                              Activation('relu')(
                              BatchNormalization()(x)))
    else:
        return lambda x: Conv3D(filters,
                         kernel_size=(1, 1, 1),
                         strides=(1, 1, 1),
                         padding=padding,
                         kernel_initializer=initializer,
                         kernel_regularizer=regularizers.l2(lbda))(
                            Activation('relu')(
                            BatchNormalization()(x)))

def relu_conv_1x1(mode, filters, initializer, lbda, padding='same'):
    if mode == '2D':
        return lambda x : Conv2D(filters,
                          kernel_size = (1, 1),
                          strides= (1, 1),
                          padding=padding,
                          kernel_initializer = initializer,
                          kernel_regularizer=regularizers.l2(lbda))(
                              Activation('relu')(x))

    else:
        return lambda x: Conv3D(filters,
                         kernel_size=(1, 1, 1),
                         strides=(1, 1, 1),
                         padding=padding,
                         kernel_initializer=initializer,
                         kernel_regularizer=regularizers.l2(lbda))(
                            Activation('relu')(x))

def conv1x1_relu(mode, filters, initializer, lbda, padding = 'same'):
    if mode == '2D':
        return lambda x : Activation('relu')(Conv2D(filters,
                                              kernel_size = (1, 1),
                                              strides= (1, 1),
                                              padding=padding,
                                              kernel_initializer = initializer,
                                              kernel_regularizer=regularizers.l2(lbda))(x))

    else:
        return lambda x: Activation('relu')(Conv3D(filters,
                                             kernel_size=(1, 1, 1),
                                             strides=(1, 1, 1),
                                             padding=padding,
                                             kernel_initializer=initializer,
                                             kernel_regularizer=regularizers.l2(lbda))(x))



def conv3x3_relu(mode, filters, initializer, lbda, padding = 'same'):
    if mode == '2D':
        return lambda x : Activation('relu')(Conv2D(filters,
                                              kernel_size = (3, 3),
                                              strides= (1, 1),
                                              padding=padding,
                                              kernel_initializer = initializer,
                                              kernel_regularizer=regularizers.l2(lbda))(x))

    else:
        return lambda x: Activation('relu')(Conv3D(filters,
                                             kernel_size=(3, 3, 3),
                                             strides=(1, 1, 1),
                                             padding=padding,
                                             kernel_initializer=initializer,
                                             kernel_regularizer=regularizers.l2(lbda))(x))



def transpose_conv3D_1x1_pr(filters, kernel_size, strides=(2, 2, 2),
                            padding="same", kernel_initializer = 'he_normal', lbda = 0):
    def t_conv_pr_instance(x):
        x = conv1x1_relu('3D', filters // 2, initializer = kernel_initializer, lbda = lbda, padding=padding)(x)
        x = Conv3DTranspose(filters= filters // 2, kernel_size=kernel_size, strides=strides,
                            padding=padding, kernel_initializer = kernel_initializer, kernel_regularizer=regularizers.l2(lbda))(x)
        x = conv1x1_relu('3D', filters, initializer = kernel_initializer, lbda = lbda, padding=padding)(x)
        return x
    return t_conv_pr_instance

'''
Dilated conv idea and aspp from deeplab paper
'''
def DilatedConv(mode, filters,  dilation_rate, initializer, lbda, padding='same'):
    #perform a dilated conv on feature map
    #Args:
    #   mode: 2D or 3D
    #   in_shape: input shape
    #   filter: num of output feature map
    #   dilation_rate: the distance between two pixels covered by the kernel
    #   lbda: param for weigth decay
    if mode == '2D':
        return lambda x : Conv2D(filters,
                                 kernel_size = (3, 3),
                                 strides = (1, 1),
                                 padding = padding,
                                 dilation_rate = dilation_rate,
                                 kernel_initializer = initializer,
                                 kernel_regularizer = regularizers.l2(lbda))(
                          Activation('relu')(
                          BatchNormalization()(x))) #remove bias regularizer, usually only weights needs to be regularized
    else:
        return lambda x : Conv3D(filters,
                                 kernel_size = (3, 3, 3),
                                 strides= (1, 1, 1),
                                 padding=padding,
                                 dilation_rate=dilation_rate,
                                 kernel_initializer = initializer,
                                 kernel_regularizer=regularizers.l2(lbda))(
                          Activation('relu')(
                          BatchNormalization()(x))) #remove bias regularizer, usually only weights needs to be regularized

def DilatedConv_no_bn(mode, filters,  dilation_rate, initializer, lbda, padding='same'):
    #perform a dilated conv on feature map
    #Args:
    #   mode: 2D or 3D
    #   in_shape: input shape
    #   filter: num of output feature map
    #   dilation_rate: the distance between two pixels covered by the kernel
    #   lbda: param for weigth decay
    if mode == '2D':
        return lambda x : Activation('relu')(Conv2D(filters,
                                             kernel_size = (3, 3),
                                             strides = (1, 1),
                                             padding = padding,
                                             dilation_rate = dilation_rate,
                                             kernel_initializer = initializer,
                                             kernel_regularizer = regularizers.l2(lbda))(x)) #remove bias regularizer, usually only weights needs to be regularized
    else:
        return lambda x : Activation('relu')(Conv3D(filters,
                                             kernel_size = (3, 3, 3),
                                             strides= (1, 1, 1),
                                             padding=padding,
                                             dilation_rate=dilation_rate,
                                             kernel_initializer = initializer,
                                             kernel_regularizer=regularizers.l2(lbda))(x)) #remove bias regularizer, usually only weights needs to be regularized

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
blocks for Merge and Run (MR) series: MRGE_net, MRGE_PR, MRGE_V2
'''

def MR_local_path(mode, filters, initializer, lbda, padding='same'):
    # implement a normal residual path in a residual block, which is used as a path in the merge and run net
    # the path is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    # bn -> relu -> conv
    if mode == '2D':
        return lambda x : Conv2D(filters,
                                 kernel_size= (3,3),
                                 strides= (1,1),
                                 padding= padding,
                                 kernel_initializer= initializer,
                                 kernel_regularizer = regularizers.l2(lbda))(
                                    Activation('relu')(BatchNormalization()(
                                    Conv2D(filters,
                                           kernel_size = (3,3),
                                           strides = (1,1),
                                           padding = padding,
                                           kernel_initializer= initializer,
                                           kernel_regularizer= regularizers.l2(lbda))(Activation('relu')(BatchNormalization()(x))))))
    else:
        return lambda x: Conv3D(filters,
                                kernel_size= (3,3,3),
                                strides=(1,1,1),
                                padding=padding,
                                kernel_initializer=initializer,
                                kernel_regularizer=regularizers.l2(lbda))(
                                    Activation('relu')(BatchNormalization()(
                                        Conv3D(filters,
                                               kernel_size=(3, 3, 3),
                                               strides=(1, 1, 1),
                                               padding=padding,
                                               kernel_initializer=initializer,
                                               kernel_regularizer=regularizers.l2(lbda))(Activation('relu')(BatchNormalization()(x))))))

def MR_global_path(mode, filters, dilation_rate, initializer, lbda, padding='same'):
    # a novel idea, to include a global path in the merge and run net implemented with dilated conv
    return lambda x : DilatedConv(mode, filters, dilation_rate, initializer, lbda, padding)(x)


def MR_local_pr(mode, filters, initializer, lbda, padding='same'):      #reduce half of the channels
    def MR_local_pr_instance(x):
        x = conv1x1_relu(mode, filters = filters // 2,  initializer=initializer, lbda = lbda, padding= padding)(x)
        x = bn_relu_conv_3x3(mode, filters=filters // 2, initializer=initializer, lbda=lbda, padding= padding)(x)
        x = bn_relu_conv_3x3(mode, filters=filters // 2, initializer=initializer, lbda=lbda, padding= padding)(x)
        out = conv1x1_relu(mode, filters=filters, initializer=initializer, lbda=lbda, padding= padding)(x)
        return out
    return MR_local_pr_instance

def MR_global_pr(mode, filters, dilation_rate, initializer, lbda, padding='same'):      #reduce half of the channels
    def MR_global_pr_instance(x):
        x = conv1x1_relu(mode, filters = filters // 2,  initializer=initializer, lbda = lbda, padding= padding)(x)
        x = DilatedConv(mode, filters = filters // 2,  dilation_rate = dilation_rate, initializer = initializer, lbda = lbda, padding=padding)(x)
        out = conv1x1_relu(mode, filters=filters, initializer=initializer, lbda=lbda, padding= padding)(x)
        return out
    return MR_global_pr_instance

def MR_local_pr_no_bn(mode, filters, initializer, lbda, padding='same'):      #reduce half of the channels
    def MR_local_pr_instance(x):
        x = conv1x1_relu(mode, filters = filters // 2,  initializer=initializer, lbda = lbda, padding= padding)(x)
        x = conv3x3_relu(mode, filters=filters // 2, initializer=initializer, lbda=lbda, padding= padding)(x)
        x = conv3x3_relu(mode, filters=filters // 2, initializer=initializer, lbda=lbda, padding= padding)(x)
        out = conv1x1_relu(mode, filters=filters, initializer=initializer, lbda=lbda, padding= padding)(x)
        return out
    return MR_local_pr_instance

def MR_global_pr_no_bn(mode, filters, dilation_rate, initializer, lbda, padding='same'):      #reduce half of the channels
    def MR_global_pr_instance(x):
        x = conv1x1_relu(mode, filters = filters // 2,  initializer=initializer, lbda = lbda, padding= padding)(x)
        x = DilatedConv_no_bn(mode, filters = filters // 2,  dilation_rate = dilation_rate, initializer = initializer, lbda = lbda, padding=padding)(x)
        out = conv1x1_relu(mode, filters=filters, initializer=initializer, lbda=lbda, padding= padding)(x)
        return out
    return MR_global_pr_instance


def MR_block(mode, filters, kernel_size, strides, initializer, lbda, padding='same'):
    # implementation of the merge-and-run block in https://arxiv.org/pdf/1611.07718.pdf
    def MR_instance(x,y):
        mid = Add()([x,y])
        x_conv = MR_local_path(mode, filters, kernel_size,  strides, initializer, lbda, padding)(x)
        y_conv = MR_local_path(mode, filters, kernel_size,  strides, initializer, lbda, padding)(y)
        x_out = Add()([x_conv,mid])
        y_out = Add()([y_conv,mid])
        return x_out, y_out

    return MR_instance


def MR_GE_block(mode, filters, dilation_rate, lbda, kernel_initializer = 'he_normal',  padding='same'):
    # GE stands for global enhanced
    # a novel idea for combining local path with global path
    def MR_instance(x,y):
        mid = Add()([x,y])
        x_conv = MR_local_path(mode, filters, kernel_initializer, lbda, padding)(x)
        y_conv = MR_global_path(mode, filters, dilation_rate, kernel_initializer, lbda, padding)(y)
        x_out = Add()([x_conv,mid])
        y_out = Add()([y_conv,mid])
        return x_out, y_out

    return MR_instance

def MR_GE_blk_pr(mode, filters, dilation_rate, lbda, kernel_initializer = 'he_normal',  padding='same'):
    # GE stands for global enhanced
    # a novel idea for combining local path with global path
    def MR_instance(x,y):
        mid = Add()([x,y])
        x_conv = MR_local_pr(mode, filters, kernel_initializer, lbda, padding)(x)
        y_conv = MR_global_pr(mode, filters, dilation_rate, kernel_initializer, lbda, padding)(y)
        x_out = Add()([x_conv,mid])
        y_out = Add()([y_conv,mid])
        return x_out, y_out
    return MR_instance

def MR_GE_blk_pr_no_bn(mode, filters, dilation_rate, lbda, kernel_initializer = 'he_normal',  padding='same'):
    # GE stands for global enhanced
    # a novel idea for combining local path with global path
    def MR_instance(x,y):
        mid = Add()([x,y])
        x_conv = MR_local_pr_no_bn(mode, filters, kernel_initializer, lbda, padding)(x)
        y_conv = MR_global_pr_no_bn(mode, filters, dilation_rate, kernel_initializer, lbda, padding)(y)
        x_out = Add()([x_conv,mid])
        y_out = Add()([y_conv,mid])
        return x_out, y_out
    return MR_instance


def MR_block_split(filters, lbda, initializer = 'he_normal', padding = 'same'):
    def MR_split_instance(x):
        x = Conv3D(filters= filters,
                   kernel_size=(1,1,1),
                   strides=(1, 1, 1),
                   padding=padding,
                   kernel_initializer= initializer,
                   kernel_regularizer=regularizers.l2(lbda))(
                         Activation('relu')(
                         BatchNormalization()(x)))
        x_out = y_out = x
        return x_out, y_out
    return MR_split_instance

def MR_block_split_no_bn(filters, lbda, initializer = 'he_normal', padding = 'same'):
    def MR_split_instance(x):
        x = Conv3D(filters= filters,
                   kernel_size=(1,1,1),
                   strides=(1, 1, 1),
                   padding=padding,
                   kernel_initializer= initializer,
                   kernel_regularizer=regularizers.l2(lbda))(x)
        x_out = y_out = x
        return x_out, y_out
    return MR_split_instance


def MR_GE_block_merge(mode, filters, dilation_rate, lbda, kernel_initializer = 'he_normal', padding='same'):
    def MR_merge_instance(x,y):
        mid = Add()([x,y])
        x_conv = MR_local_path(mode, filters, kernel_initializer, lbda, padding)(x)
        y_conv = MR_global_path(mode, filters, dilation_rate, kernel_initializer, lbda, padding)(y)
        out = Add()([Add()([x_conv, y_conv]), mid])
        return out
    return MR_merge_instance

def MR_GE_block_merge_pr(mode, filters, dilation_rate, lbda, kernel_initializer = 'he_normal', padding='same'):
    def MR_merge_instance(x,y):
        mid = Add()([x,y])
        x_conv = MR_local_pr(mode, filters, kernel_initializer, lbda, padding)(x)
        y_conv = MR_global_pr(mode, filters, dilation_rate, kernel_initializer, lbda, padding)(y)
        out = Add()([Add()([x_conv, y_conv]), mid])
        return out
    return MR_merge_instance

def MR_GE_block_merge_no_bn(mode, filters, dilation_rate, lbda, kernel_initializer = 'he_normal', padding='same'):
    def MR_merge_instance(x,y):
        mid = Add()([x,y])
        x_conv = MR_local_pr_no_bn(mode, filters, kernel_initializer, lbda, padding)(x)
        y_conv = MR_global_pr_no_bn(mode, filters, dilation_rate, kernel_initializer, lbda, padding)(y)
        out = Add()([Add()([x_conv, y_conv]), mid])
        return out
    return MR_merge_instance

def MRGE_exp_block(mode, filters, dilation_max, lbda):
    def MRGE_exp_instance(x):
        x, y = MR_block_split(filters, lbda)(x)
        block_num = int(log2(dilation_max) + 1)
        rate_list = [2 ** i for i in range(block_num)]
        for rate in rate_list[:-1]:
            x, y = MR_GE_block(mode, filters=filters, dilation_rate=rate, lbda=lbda)(x, y)
        x = MR_GE_block_merge(mode, filters=filters, dilation_rate=rate_list[-1], lbda=lbda)(x, y)
        return x
    return MRGE_exp_instance

def MRGE_exp_blk_pr(mode, filters, dilation_max, lbda):
    def MRGE_exp_pr_instance(x):
        x, y = MR_block_split(filters, lbda)(x)
        block_num = int(log2(dilation_max) + 1)
        rate_list = [2 ** i for i in range(block_num)]
        for rate in rate_list[:-1]:
            x, y = MR_GE_blk_pr(mode, filters=filters, dilation_rate=rate, lbda=lbda)(x, y)
        x = MR_GE_block_merge(mode, filters=filters, dilation_rate=rate_list[-1], lbda=lbda)(x, y)
        return x
    return MRGE_exp_pr_instance

def MRGE_exp_blk_pr_no_bn(mode, filters, dilation_max, lbda):
    def MRGE_exp_pr_instance(x):
        x, y = MR_block_split_no_bn(filters, lbda)(x)
        block_num = int(log2(dilation_max) + 1)
        rate_list = [2 ** i for i in range(block_num)]
        for rate in rate_list[:-1]:
            x, y = MR_GE_blk_pr_no_bn(mode, filters=filters, dilation_rate=rate, lbda=lbda)(x, y)
        x = MR_GE_block_merge_no_bn(mode, filters=filters, dilation_rate=rate_list[-1], lbda=lbda)(x, y)
        return x
    return MRGE_exp_pr_instance

"""
blocks for 2-cardinal MRGE(local enhanced):
"""
def MR_loc_en_pr(mode, filters, initializer, lbda, padding='same'):      #reduce half of the channels
    def MR_loc_en_pr_instance(x):
        x_f3 = x_f5 = x
        x_f3 = conv1x1_relu(mode, filters = filters // 4,  initializer=initializer, lbda = lbda, padding= padding)(x_f3)
        x_f3 = bn_relu_conv_3x3(mode, filters=filters // 4, initializer=initializer, lbda=lbda, padding=padding)(x_f3)
        x_f3 = conv1x1_relu(mode, filters=filters, initializer=initializer, lbda=lbda, padding=padding)(x_f3)
        x_f5 = conv1x1_relu(mode, filters = filters // 4,  initializer=initializer, lbda = lbda, padding= padding)(x_f5)
        x_f5 = bn_relu_conv_3x3(mode, filters=filters // 4, initializer=initializer, lbda=lbda, padding= padding)(x_f5)
        x_f5 = bn_relu_conv_3x3(mode, filters=filters // 4, initializer=initializer, lbda=lbda, padding= padding)(x_f5)
        x_f5 = conv1x1_relu(mode, filters=filters, initializer=initializer, lbda=lbda, padding= padding)(x_f5)
        out = Add()([x_f3, x_f5])
        return out
    return MR_loc_en_pr_instance

def MR_GELE_pr(mode, filters, dilation_rate, lbda, kernel_initializer = 'he_normal',  padding='same'):
    # GE stands for global enhanced
    # a novel idea for combining local path with global path
    def MR_instance(x,y):
        mid = Add()([x,y])
        x_conv = MR_loc_en_pr(mode, filters, kernel_initializer, lbda, padding)(x)
        y_conv = MR_global_pr(mode, filters, dilation_rate, kernel_initializer, lbda, padding)(y)
        x_out = Add()([x_conv,mid])
        y_out = Add()([y_conv,mid])
        return x_out, y_out
    return MR_instance

def MR_GELE_merge_pr(mode, filters, dilation_rate, lbda, kernel_initializer = 'he_normal', padding='same'):
    def MR_merge_instance(x,y):
        mid = Add()([x,y])
        x_conv = MR_loc_en_pr(mode, filters, kernel_initializer, lbda, padding)(x)
        y_conv = MR_global_pr(mode, filters, dilation_rate, kernel_initializer, lbda, padding)(y)
        out = Add()([Add()([x_conv, y_conv]), mid])
        return out
    return MR_merge_instance

def MRGE_LE_exp_blk_pr(mode, filters, dilation_max, lbda):
    def MRGE_exp_pr_instance(x):
        x, y = MR_block_split(filters, lbda)(x)
        block_num = int(log2(dilation_max) + 1)
        rate_list = [2 ** i for i in range(block_num)]
        for rate in rate_list[:-1]:
            x, y = MR_GELE_pr(mode, filters=filters, dilation_rate=rate, lbda=lbda)(x, y)
        x = MR_GELE_merge_pr(mode, filters=filters, dilation_rate=rate_list[-1], lbda=lbda)(x, y)
        return x
    return MRGE_exp_pr_instance

'''
def MRGE_exp_channel_reduced(mode, filters, dilation_max, lbda):
    def MRGE_exp_cr_instance(x):
        fin = filters // 4
        x, y = MR_block_split(fin, lbda)(x)
        block_num = int(log2(dilation_max) + 1)
        rate_list = [2 ** i for i in range(block_num)]
        for rate in rate_list[:-1]:
            x, y = MR_GE_block(mode, filters=fin, dilation_rate=rate, lbda=lbda)(x, y)
        x = MR_GE_block_merge(mode, filters=fin, dilation_rate=rate_list[-1], lbda=lbda)(x, y)

        x = Conv3D(filters= filters,
                   kernel_size=(1,1,1),
                   strides=(1, 1, 1),
                   padding='same',
                   kernel_initializer= 'he_normal',
                   kernel_regularizer=regularizers.l2(lbda))(
                         Activation('relu')(
                 BatchNormalization()(x)))

        return x
    return MRGE_exp_cr_instance


def MRGE_inc_channel_reduced(mode, filters, dilation_max, lbda):
    def MRGE_inc_cr_instance(x):
        fin = filters // 2
        x, y = MR_block_split(fin, lbda)(x)
        for rate in range(dilation_max)[:-1]:
            x, y = MR_GE_block(mode, filters=fin, dilation_rate=(rate+1), lbda=lbda)(x, y)
        x = MR_GE_block_merge(mode, filters=fin, dilation_rate=dilation_max, lbda=lbda)(x, y)
        x = Conv3D(filters=filters,
                   kernel_size=(1, 1, 1),
                   strides=(1, 1, 1),
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(lbda))(
                Activation('relu')(
                BatchNormalization()(x)))

        return x

    return MRGE_inc_cr_instance

'''



"""
blocks for DCCN series: DCCN_ORI, DCCN_SYMM
"""

def denseBlock(mode, l, k, lbda):
    if mode == '2D':
        def dense_block_instance(x):
            ins = [x, denseConv('2D',k,3,lbda)(
                      denseConv('2D',k,1, lbda)(x))]
            for i in range(l-1):
                ins.append(denseConv('2D',k,3, lbda)(
                           denseConv('2D',k,1, lbda)(Concatenate(axis=-1)(ins))))
            y = Concatenate(axis=-1)(ins)
            return y
        return dense_block_instance
    else:
        def dense_block_instance(x):
            ins = [x, denseConv('3D',k,3, lbda)(
                      denseConv('3D',k,1, lbda)(x))]
            for i in range(l-1):
                ins.append(denseConv('3D',k,3, lbda)(
                           denseConv('3D',k,1, lbda)(Concatenate(axis=-1)(ins))))
            y = Concatenate(axis=-1)(ins)
            return y
        return dense_block_instance

def denseConv(mode, k, kernel_size, lbda):
    if mode == '2D':
        return lambda x: Conv2D(filters=k,
                                kernel_size=2*(kernel_size,),
                                padding='same',
                                kernel_regularizer=regularizers.l2(lbda),
                                bias_regularizer=regularizers.l2(lbda))(
                         Activation('relu')(
                         BatchNormalization()(x)))
    else:
        return lambda x: Conv3D(filters=k,
                                kernel_size=3*(kernel_size,),
                                padding='same',
                                kernel_regularizer=regularizers.l2(lbda),
                                bias_regularizer=regularizers.l2(lbda))(
                         Activation('relu')(
                         BatchNormalization()(x)))

# Transition Layers
def transitionLayerPool(mode, f, lbda):
    if mode == '2D':
        return lambda x: AveragePooling2D(pool_size=2*(2,))(
                         denseConv('2D', f, 1, lbda)(x))
    else:
        return lambda x: AveragePooling3D(pool_size=3*(2,))(
                         denseConv('3D', f, 1, lbda)(x))

def transitionLayerTransposeUp(mode, f, lbda):
    if mode == '2D':
        return lambda x: Conv2DTranspose(filters=f, kernel_size=(3, 3), strides=(2, 2),
                                         padding="same", kernel_regularizer = regularizers.l2(lbda))(
                         denseConv('2D', f, 1, lbda)(x))
    else:
        return lambda x: Conv3DTranspose(filters=f, kernel_size=(3, 3, 3), strides=(2, 2, 2),
                                         padding="same", kernel_regularizer = regularizers.l2(lbda))(
                         denseConv('3D', f, 1, lbda)(x))

class resize_2D(Layer):

    def __init__(self, out_res=24, **kwargs):
        self.input_dim = None
        self.out_res = out_res
        super(resize_2D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_dim = input_shape[1:]
        super(resize_2D, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        size=(K.constant(np.array(2*(self.out_res,), dtype=np.int32), dtype=K.tf.int32))
        y = K.tf.image.resize_bilinear(images=x, size=size)
        return y

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + 2 * (self.out_res,) + (input_shape[-1],)

class resize_3D(Layer):

    def __init__(self, out_res=24, **kwargs):
        self.input_dim = None
        self.out_res = out_res
        super(resize_3D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_dim = input_shape[1:]
        super(resize_3D, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        y = K.reshape(x=x,
                      shape=(-1,
                             self.input_dim[0],
                             self.input_dim[1],
                             self.input_dim[2] * self.input_dim[3]))
        y = K.tf.image.resize_bilinear(images=y,
                                       size=(K.constant(np.array(2*(self.out_res,),
                                                                 dtype=np.int32),
                                                        dtype=K.tf.int32)))
        y = K.reshape(x=y,
                      shape=(-1,
                             self.out_res,
                             self.out_res,
                             self.input_dim[2],
                             self.input_dim[3]))
        y = K.permute_dimensions(x=y, pattern=(0,1,3,2,4))
        y = K.reshape(x=y,
                      shape=(-1,
                             self.out_res,
                             self.input_dim[2],
                             self.out_res * self.input_dim[3]))
        y = K.tf.image.resize_bilinear(images=y,
                                       size=(K.constant(np.array(2*(self.out_res,),
                                                                 dtype=np.int32),
                                                       dtype=K.tf.int32)))
        y = K.reshape(x=y,
                      shape=(-1,
                             self.out_res,
                             self.out_res,
                             self.out_res,
                             self.input_dim[3]))
        y = K.permute_dimensions(x=y, pattern=(0,1,3,2,4))
        return y

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + 3 * (self.out_res,) + (input_shape[-1],)

