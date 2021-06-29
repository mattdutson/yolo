## Overview

This project aims to be a simple, clean implementation of [YOLOv3](https://arxiv.org/abs/1804.02767) in TensorFlow 2. It is mostly designed for inference using the original author's pre-trained weights. However, with some minor modifications it could also be used for training.

## Scripts

The `scripts` folder contains the following:
- `detect.py`: detects objects in an image and draws bounding box annotations
- `convert.py`: converts pre-trained weights from the custom `.weights` format to a standard format (Keras `.h5` or TensorFlow `.ckpt`)
  
Use the `--help` flag to see script options and usage patterns.

Scripts assume that the current directory is on `PYTHONPATH`. If this is not the case, you will get a `ModuleNotFoundError`. In Bash, the command to add the current directory to `PYTHONPATH` is:
```bash
export PYTHONPATH=".:$PYTHONPATH"
```

Scripts also assume they're being run from the repository's top-level directory. So, an invocation looks like:
```bash
./scripts/detect.py <ARGUMENTS>
```

## Python

The `yolo.models` module contains one function, `yolo_v3`, that returns a Keras model. The optional `darknet_weights` parameter specifies a `.weights` file containing pretrained weights. The `n_classes` parameter sets the number of output classes (the default is the 80 [COCO classes](https://cocodataset.org/#explore)).

The `yolo.utils` module contains the following functions:
- `annotate`: draws bounding box annotations on an image
- `postprocess`: converts raw model outputs into bounding box predictions by applying a non-max suppression algorithm
- `preprocess`: prepares an RGB uint8 image for inference
- `read_names`: reads a text file line by line to extract class names (see `names/coco.txt`)

####

## Conda Environment

To create the `yolo` environment, run:
```
conda env create -f conda/environment.yml
```

To enable GPU support, instead run:
```
conda env create -f conda/environment_gpu.yml
```

## Code Style

Unless otherwise specified, follow the [PEP8](https://www.python.org/dev/peps/pep-0008) conventions.

Use a line limit of 99 characters.
