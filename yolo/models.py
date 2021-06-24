import tensorflow as tf
from tensorflow.keras.layers import *


def yolo_v3(darknet_weights=None):
    inputs = Input(shape=(None, None, 3))
    x = inputs

    x = _conv_block(x, 32)

    x = _conv_block(x, 64, strides=2)
    for _ in range(1):
        x = _residual_block(x, 64)

    x = _conv_block(x, 128, strides=2)
    for _ in range(2):
        x = _residual_block(x, 128)

    x = _conv_block(x, 256, strides=2)
    for _ in range(8):
        x = _residual_block(x, 256)

    x = _conv_block(x, 512, strides=2)
    for _ in range(8):
        x = _residual_block(x, 512)

    x = _conv_block(x, 1024, strides=2)
    for _ in range(4):
        x = _residual_block(x, 1024)

    return tf.keras.Model(inputs=inputs, outputs=x)


def _conv_block(x, filters, kernel_size=3, strides=1):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x


def _residual_block(x, filters):
    skip = x
    x = _conv_block(x, filters // 2, kernel_size=1)
    x = _conv_block(x, filters)
    x = Add()([skip, x])
    return x
