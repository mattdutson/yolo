import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *


# Thanks to github.com/zzh8829 for information on how to read
# Darknet weight files. I use a few lines of their weight-loading code
# here. Source link:
# https://github.com/zzh8829/yolov3-tf2/blob/master/yolov3_tf2/utils.py


def yolo_v3(darknet_weights=None):
    inputs = Input(shape=(None, None, 3))
    x = inputs

    if darknet_weights is not None:
        w = open(darknet_weights, 'rb')
        np.fromfile(w, dtype=np.int32, count=5)  # File header
    else:
        w = None

    x = _conv_block(x, w, 32)

    x = _conv_block(x, w, 64, strides=2)
    for _ in range(1):
        x = _residual_block(x, w, 64)

    x = _conv_block(x, w, 128, strides=2)
    for _ in range(2):
        x = _residual_block(x, w, 128)

    x = _conv_block(x, w, 256, strides=2)
    for _ in range(8):
        x = _residual_block(x, w, 256)

    x = _conv_block(x, w, 512, strides=2)
    for _ in range(8):
        x = _residual_block(x, w, 512)

    x = _conv_block(x, w, 1024, strides=2)
    for _ in range(4):
        x = _residual_block(x, w, 1024)

    return tf.keras.Model(inputs=inputs, outputs=x)


def _conv_block(x, w, filters, kernel_size=3, strides=1):
    conv = Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)
    x = conv(x)
    bn = BatchNormalization()
    x = bn(x)
    x = LeakyReLU(alpha=0.1)(x)

    if w is not None:
        # TF stores BN weights as [gamma, beta, mean, variance]. Darknet
        # stores them as [beta, gamma, mean, variance].
        w_bn = np.fromfile(w, dtype=np.float32, count=4 * filters)
        w_bn = w_bn.reshape((4, filters))[[1, 0, 2, 3]]
        bn.set_weights(w_bn)

        # A TF kernel has shape (h, w, c_in, c_out). A Darknet kernel
        # has shape (c_out, c_in, h, w).
        s_tf = conv.kernel.shape
        s_darknet = (s_tf[3], s_tf[2], s_tf[0], s_tf[1])
        w_conv = np.fromfile(w, dtype=np.float32, count=np.prod(s_darknet))
        w_conv = w_conv.reshape(s_darknet).transpose([2, 3, 1, 0])
        conv.set_weights([w_conv])

    return x


def _residual_block(x, w, filters):
    skip = x
    x = _conv_block(x, w, filters // 2, kernel_size=1)
    x = _conv_block(x, w, filters)
    x = Add()([skip, x])
    return x
