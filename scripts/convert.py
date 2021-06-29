#!/usr/bin/env python3

import os
import os.path as path
from argparse import ArgumentParser

from yolo.models import yolo_v3


def main(args):
    model = yolo_v3(darknet_weights=args.darknet_weights)
    os.makedirs(path.dirname(args.output_file), exist_ok=True)
    model.save_weights(args.output_file)


def parse_args():
    parser = ArgumentParser()

    # Required arguments
    parser.add_argument('darknet_weights',
                        help='a Darknet .weights file')
    parser.add_argument('output_file',
                        help='location to save output weights')

    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
