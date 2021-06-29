#!/usr/bin/env python3

import os
import os.path as path
from argparse import ArgumentParser

import cv2 as cv
import tensorflow as tf

from yolo.models import yolo_v3
from yolo.utils import annotate, postprocess, preprocess, read_names


def main(args):
    model = yolo_v3(darknet_weights=args.darknet_weights)
    names = read_names(args.names_file)
    image_bgr = cv.imread(args.input_file)
    image_rgb = preprocess(tf.reverse(image_bgr, [-1]))
    boxes, classes = model(image_rgb)
    results = postprocess(boxes[0], classes[0],
                          max_boxes=args.max_boxes, iou_threshold=args.iou_threshold,
                          score_threshold=args.score_threshold)
    annotated = annotate(image_bgr, *results, names, color=args.annotate_color)
    os.makedirs(path.dirname(args.output_file), exist_ok=True)
    cv.imwrite(args.output_file, annotated)


def parse_args():
    parser = ArgumentParser()

    # Required arguments
    parser.add_argument('darknet_weights',
                        help='a Darknet .weights file')
    parser.add_argument('input_file',
                        help='an input image')
    parser.add_argument('output_file',
                        help='location to save output image')

    # Optional arguments
    parser.add_argument('-a', '--annotate-color',
                        default=[255, 0, 0], nargs=3, type=int,
                        help='RGB color for annotations')
    parser.add_argument('-c', '--score-threshold',
                        default=0.7, type=float,
                        help='confidence threshold (0-1)')
    parser.add_argument('-i', '--iou-threshold',
                        default=0.4, type=float,
                        help='maximum allowed box overlap (0-1)')
    parser.add_argument('-m', '--max-boxes',
                        default=100, type=int,
                        help='maximum number of boxes to detect')
    parser.add_argument('-n', '--names-file',
                        default=path.join('names', 'coco.txt'),
                        help='file containing class names')

    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
