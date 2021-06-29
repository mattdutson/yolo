import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *

# Resources consulted:
# https://arxiv.org/abs/1804.02767
# https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
# https://github.com/zzh8829/yolov3-tf2/blob/master/yolov3_tf2/models.py
# https://github.com/zzh8829/yolov3-tf2/blob/master/yolov3_tf2/utils.py


LEAK_ALPHA = 0.1
ANCHORS_1 = [(116, 90), (156, 198), (373, 326)]
ANCHORS_2 = [(30, 61), (62, 45), (59, 119)]
ANCHORS_3 = [(10, 13), (16, 30), (33, 23)]


def yolo_v3(n_classes=80, darknet_weights=None):
    inputs = Input(shape=(None, None, 3))
    x = inputs

    if darknet_weights is not None:
        w = open(darknet_weights, 'rb')
        np.fromfile(w, dtype=np.int32, count=5)  # Skip file header
    else:
        w = None

    # Darknet backbone
    x = _conv_block(x, w, 32)
    x = _conv_block(x, w, 64, strides=2)
    for _ in range(1):
        x = _residual_block(x, w, 32)
    x = _conv_block(x, w, 128, strides=2)
    for _ in range(2):
        x = _residual_block(x, w, 64)
    x = _conv_block(x, w, 256, strides=2)
    for _ in range(8):
        x = _residual_block(x, w, 128)
    skip_36 = x
    x = _conv_block(x, w, 512, strides=2)
    for _ in range(8):
        x = _residual_block(x, w, 256)
    skip_61 = x
    x = _conv_block(x, w, 1024, strides=2)
    for _ in range(4):
        x = _residual_block(x, w, 512)

    # Output block 1
    x, boxes_1, classes_1 = _output_block(x, w, 512, n_classes, ANCHORS_1, scale=32)

    # Output block 2
    x = _conv_block(x, w, 256, kernel_size=1)
    x = UpSampling2D(size=2)(x)
    x = Concatenate()([x, skip_61])
    x, boxes_2, classes_2 = _output_block(x, w, 256, n_classes, ANCHORS_2, scale=16)

    # Output block 3
    x = _conv_block(x, w, 128, kernel_size=1)
    x = UpSampling2D(size=2)(x)
    x = Concatenate()([x, skip_36])
    x, boxes_3, classes_3 = _output_block(x, w, 128, n_classes, ANCHORS_3, scale=8)

    if darknet_weights is not None:
        w.close()

    boxes = tf.concat([boxes_1, boxes_2, boxes_3], axis=1)
    classes = tf.concat([classes_1, classes_2, classes_3], axis=1)
    return tf.keras.Model(inputs=inputs, outputs=[boxes, classes])


def _conv_block(x, w, filters, kernel_size=3, strides=1, batch_norm=True):
    # Thanks to github.com/zzh8829 for this padding logic. I never
    # would have guessed the top-left padding, it seems very arbitrary!
    if strides == 1:
        padding = 'same'
    else:
        padding = 'valid'
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)

    conv = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=not batch_norm)
    x = conv(x)
    if batch_norm:
        bn = BatchNormalization()
        x = bn(x)
    else:
        bn = None
    x = LeakyReLU(alpha=LEAK_ALPHA)(x)

    # More thanks to github.com/zzh8829 for information on how to read
    # Darknet weight files. I use a few lines of their weight-loading
    # code here.
    if w is not None:
        if batch_norm:
            # Darknet uses order [beta, gamma, mean, variance].
            w_bn = np.fromfile(w, dtype=np.float32, count=4 * filters)
            w_bn = w_bn.reshape((4, filters))[[1, 0, 2, 3]]
            bn.set_weights(w_bn)
        else:
            b_conv = np.fromfile(w, dtype=np.float32, count=filters)
            conv.bias.assign(b_conv)

        # A Darknet kernel has shape (c_out, c_in, h, w).
        shape_tf = conv.kernel.shape
        shape_darknet = (shape_tf[3], shape_tf[2], shape_tf[0], shape_tf[1])
        w_conv = np.fromfile(w, dtype=np.float32, count=np.prod(shape_darknet))
        w_conv = w_conv.reshape(shape_darknet).transpose([2, 3, 1, 0])
        conv.kernel.assign(w_conv)

    return x


def _output_block(x, w, filters, n_classes, anchors, scale):
    for _ in range(2):
        x = _conv_block(x, w, filters, kernel_size=1)
        x = _conv_block(x, w, filters * 2)
    x = _conv_block(x, w, filters, kernel_size=1)
    skip = x
    x = _conv_block(x, w, filters * 2)
    x = _conv_block(x, w, len(anchors) * (n_classes + 5), kernel_size=1, batch_norm=False)

    boxes = []
    classes = []
    for i, anchor in enumerate(anchors):
        # See Section 2.1 of https://arxiv.org/abs/1804.02767.
        offset = i * (n_classes + 5)
        grid_x, grid_y = tf.meshgrid(tf.range(tf.shape(x)[2], dtype=x.dtype),
                                     tf.range(tf.shape(x)[1], dtype=x.dtype))
        box_x = scale * (tf.sigmoid(x[..., offset + 0]) + grid_x)
        box_y = scale * (tf.sigmoid(x[..., offset + 1]) + grid_y)
        box_w = tf.exp(x[..., offset + 2]) * anchor[0]
        box_h = tf.exp(x[..., offset + 3]) * anchor[1]
        class_scores = tf.sigmoid(x[..., offset + 5: offset + 5 + n_classes])
        box_score = tf.sigmoid(x[..., offset + 4]) * tf.reduce_max(class_scores, axis=-1)

        box_x_1 = box_x - box_w / 2
        box_x_2 = box_x + box_w / 2
        box_y_1 = box_y - box_h / 2
        box_y_2 = box_y + box_h / 2
        boxes_i = tf.stack([box_y_1, box_x_1, box_y_2, box_x_2, box_score], axis=-1)
        boxes_i = tf.reshape(boxes_i, (tf.shape(boxes_i)[0], -1, boxes_i.shape[-1]))
        boxes.append(boxes_i)

        classes_i = tf.argmax(class_scores, axis=-1)
        classes_i = tf.reshape(classes_i, (tf.shape(classes_i)[0], -1))
        classes.append(classes_i)

    boxes = tf.concat(boxes, axis=1)
    classes = tf.concat(classes, axis=1)
    return skip, boxes, classes


def _residual_block(x, w, filters):
    skip = x
    x = _conv_block(x, w, filters, kernel_size=1)
    x = _conv_block(x, w, filters * 2)
    x = Add()([skip, x])
    return x
