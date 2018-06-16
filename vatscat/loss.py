from keras import backend as K
import tensorflow as tf

'''
binary classification class focal loss
'''
def binary_focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1 , gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

'''
Multi class focal loss
'''
def focal_loss(gamma = 2., alpha = .25, cls = 4):
    def focal_loss_instance(y_true, y_pred):
        y_hardmax = K.one_hot(K.argmax(y_pred, axis = -1), cls)
        pt_0 = tf.where(tf.equal(y_true[:, :, 0], 1), y_hardmax[:, :, 0], tf.ones_like(y_hardmax[:, :, 0]))
        pt_1 = tf.where(tf.equal(y_true[:, :, 1], 1), y_hardmax[:, :, 1], tf.ones_like(y_hardmax[:, :, 1]))
        pt_2 = tf.where(tf.equal(y_true[:, :, 2], 1), y_hardmax[:, :, 2], tf.ones_like(y_hardmax[:, :, 2]))
        pt_3 = tf.where(tf.equal(y_true[:, :, 3], 1), y_hardmax[:, :, 3], tf.ones_like(y_hardmax[:, :, 3]))
        return -K.sum(alpha * K.pow(1. - pt_0, gamma) * K.log(pt_0)) - K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - \
               K.sum(alpha * K.pow(1. - pt_2, gamma) * K.log(pt_2)) - K.sum(alpha * K.pow(1. - pt_3, gamma) * K.log(pt_3))