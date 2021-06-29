import cv2 as cv
import tensorflow as tf


def annotate(image, x_1, x_2, y_1, y_2, scores, classes, names, color=(255, 0, 0)):
    color_bgr = tuple(reversed(color))
    for x_1_i, x_2_i, y_1_i, y_2_i, class_, score in zip(x_1, x_2, y_1, y_2, classes, scores):
        x_1_i, x_2_i, y_1_i, y_2_i = map(int, [x_1_i, x_2_i, y_1_i, y_2_i])
        image = cv.rectangle(image, (x_1_i, y_1_i), (x_2_i, y_2_i), color=color_bgr, thickness=2)
        text = '{} - {:.2f}%'.format(names[class_].capitalize(), score * 100)
        image = cv.putText(image, text, (x_1_i, y_1_i - 5), cv.FONT_HERSHEY_SIMPLEX,
                           fontScale=0.5, color=color_bgr)
    return image


def postprocess(boxes, classes, max_boxes=100, iou_threshold=0.4, score_threshold=0.7):
    selected = tf.image.non_max_suppression(boxes[..., :4], boxes[..., 4], max_boxes,
                                            iou_threshold=iou_threshold,
                                            score_threshold=score_threshold)
    selected = selected.numpy()
    boxes = boxes.numpy()[selected]
    y_1, x_1, y_2, x_2, scores = [boxes[:, i] for i in range(5)]
    classes = classes.numpy()[selected]
    return x_1, x_2, y_1, y_2, scores, classes


def preprocess(image):
    pad_h = -image.shape[0] % 32
    pad_w = -image.shape[1] % 32
    image = tf.pad(image, tf.constant([[0, pad_h], [0, pad_w], [0, 0]]))
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.expand_dims(image, axis=0)
    return image


def read_names(filename):
    with open(filename, 'r') as f:
        names = f.readlines()
        names = list(map(str.strip, names))
        return names
