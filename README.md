# Age and Gender Estimation
This is a Keras implementation of a CNN for estimating age and gender from a face image.
In training, [the IMDB-WIKI dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) is used.

## Dependencies
- Python3.6+

Tested on:
- Windows 10, Python 3.6.9, Tensorflow 2.3.0, CUDA 10.01, cuDNN 7.6

## Setup
- Install Anaconda
- Open Anaconda Prompt and run following commands
    - cd age-estimator
    - 

## Usage

### Use trained model for demo
Run the demo script (requires web cam).
You can use `--image_dir [IMAGE_DIR]` option to use images in the `[IMAGE_DIR]` directory instead.

```sh
python app.py
```

```sh
usage: demo.py [-h] [--weight_file WEIGHT_FILE] [--margin MARGIN]
               [--image_dir IMAGE_DIR]

This script detects faces from web cam input, and estimates age and gender for
the detected faces.

optional arguments:
  -h, --help            show this help message and exit
  --weight_file WEIGHT_FILE
                        path to weight file (e.g. weights.28-3.73.hdf5)
                        (default: None)
  --margin MARGIN       margin around detected face for age-gender estimation
                        (default: 0.4)
  --image_dir IMAGE_DIR
                        target image directory; if set, images in image_dir
                        are used instead of webcam (default: None)
```