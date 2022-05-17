"""
app.py:
    version 0.0.1

    A python file/module which contains the functions to
	show the demo (demonstration) for `Age Estimator` solution.

    - Step 1: Create python environment from `setup\requirements.txt`
    - Step 2: Run `python app.py`
"""

# Importing the required modules/packages
from pathlib import Path
import cv2
import dlib
import numpy as np
import argparse
from contextlib import contextmanager
from omegaconf import OmegaConf
from tensorflow.keras.utils import get_file
from utils.model_utils import get_model

# Mod hash for pretrained model
modhash = "6d7f7b7ced093a8b3ef6399163da6ece"

# Path of pre-trained model
pretrained_model = r"./pretrained_models/EfficientNetB3_224_weights.11-3.44.hdf5"


def get_args():
    """
    Function to get/extract the required arguments

    Parameters:
    -----------
    NA

    Returns:
    --------
    args: `argparse.ArgumentParser.parse_args`
        A valid set of reuired arguments

    References:
    -----------
    NA

    Examples:
    ---------
    get_args()
    """

    # Creating object of ArgumentParser
    parser = argparse.ArgumentParser(
        description="This script detects faces from web cam input, "
        "and estimates age and gender for the detected faces.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Adding weight_file details
    parser.add_argument(
        "--weight_file",
        type=str,
        default=pretrained_model,
        help="path to weight file (e.g. weights.28-3.73.hdf5)",
    )

    # Adding margin details
    parser.add_argument(
        "--margin",
        type=float,
        default=0.4,
        help="margin around detected face for age-gender estimation",
    )

    # Adding image_dir details
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="target image directory; if set, images in image_dir are used instead of webcam",
    )
    args = parser.parse_args()
    return args


def draw_label(
    image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8, thickness=1
):
    """
    Function to get/extract the required arguments

    Parameters:
    -----------
    image: `OpenCV image object`
        A valid python `OpenCV image object` containing
        image details.

    point: `list` or `tuple`
        A valid python `list` or `tuple` containing X and Y
        values for pixcel where label needs to draw.

    label: `str`
        A valid python `str` containing label details.

    font: `OpenCV Font Name`
        A valid python `OpenCV Font Name` containing font name for
        the label which needs to draw.

    font_scale: `OpenCV Font Scale`
        A valid python `OpenCV Font Scale` containing font scale for
        the label which needs to draw.

    thickness: `OpenCV Thickness`
        A valid python `OpenCV Thickness` containing thickness for
        the label which needs to draw.

    Returns:
    --------
    NA

    References:
    -----------
    NA

    Examples:
    ---------
    draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8, thickness=1)
    """

    # Creating text size
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]

    # Getting X and Y values for pixcel where label needs to draw
    x, y = point

    # Drawing filled rectange for label
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)

    # Drawing label
    cv2.putText(
        image,
        label,
        point,
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )


@contextmanager
def video_capture(*args, **kwargs):
    """
    Function to capture/yield the video

    Parameters:
    -----------
    args:
        A valid python arguments.

    kwargs: `dict` or `tuple`
        A valid python `dict` of arguments.

    Returns:
    --------
    NA

    References:
    -----------
    NA

    Examples:
    ---------
    video_capture(0)
    """

    # Creating object of VideoCapture
    cap = cv2.VideoCapture(*args, **kwargs)

    try:
        # Yielding cap (VideoCapture object)
        yield cap
    finally:
        # Releasing cap (VideoCapture object)
        cap.release()


def yield_images():
    """
    Function to yield the images

    Parameters:
    -----------
    NA

    Returns:
    --------
    NA

    References:
    -----------
    NA

    Examples:
    ---------
    yield_images()
    """
    # capture video
    with video_capture(0) as cap:
        # Setting frame width
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # Setting frame height
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            # get video frame
            ret, img = cap.read()

            if not ret:
                raise RuntimeError("Failed to capture image")

            yield img


def yield_images_from_dir(image_dir):
    """
    Function to yield the images from a given
    image directory.

    Parameters:
    -----------
    image_dir: `str`
        A valid python `str` containing path of image directory.
        The image directory contains already saved images.

    Returns:
    --------
    NA

    References:
    -----------
    NA

    Examples:
    ---------
    yield_images_from_dir(image_dir)
    """

    # Creating Path object of image directory
    image_dir = Path(image_dir)

    # Iterating over each file of image directory
    for image_path in image_dir.glob("*.*"):

        # Reading the image
        img = cv2.imread(str(image_path), 1)

        if img is not None:
            # Reshaping the image
            h, w, _ = img.shape
            r = 640 / max(w, h)
            yield cv2.resize(img, (int(w * r), int(h * r)))


def main():
    """
    Function to handle the execution pipeline.

    Parameters:
    -----------
    NA

    Returns:
    --------
    NA

    References:
    -----------
    NA

    Examples:
    ---------
    yield_images_from_dir(image_dir)
    """

    # Getting the required arguments
    args = get_args()

    # Getting teh weight_file
    weight_file = args.weight_file

    # Getting the margin
    margin = args.margin

    # Getting the image directory
    image_dir = args.image_dir

    if not weight_file:
        weight_file = get_file(
            "EfficientNetB3_224_weights.11-3.44.hdf5",
            pretrained_model,
            cache_subdir="pretrained_models",
            file_hash=modhash,
            cache_dir=str(Path(__file__).resolve().parent),
        )

    # for face detection
    detector = dlib.get_frontal_face_detector()

    # load model and weights
    model_name, img_size = Path(weight_file).stem.split("_")[:2]
    img_size = int(img_size)
    cfg = OmegaConf.from_dotlist(
        [f"model.model_name={model_name}", f"model.img_size={img_size}"]
    )
    model = get_model(cfg)
    model.load_weights(weight_file)

    image_generator = yield_images_from_dir(image_dir) if image_dir else yield_images()

    for img in image_generator:
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)

        # detect faces using dlib detector
        detected = detector(input_img, 1)
        faces = np.empty((len(detected), img_size, img_size, 3))

        if len(detected) > 0:
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = (
                    d.left(),
                    d.top(),
                    d.right() + 1,
                    d.bottom() + 1,
                    d.width(),
                    d.height(),
                )
                xw1 = max(int(x1 - margin * w), 0)
                yw1 = max(int(y1 - margin * h), 0)
                xw2 = min(int(x2 + margin * w), img_w - 1)
                yw2 = min(int(y2 + margin * h), img_h - 1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                faces[i] = cv2.resize(
                    img[yw1 : yw2 + 1, xw1 : xw2 + 1], (img_size, img_size)
                )

            # predict ages and genders of the detected faces
            results = model.predict(faces)
            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()

            # draw results
            for i, d in enumerate(detected):
                label = "{}".format(
                    int(predicted_ages[i]),
                    "M" if predicted_genders[i][0] < 0.5 else "F",
                )
                draw_label(img, (d.left(), d.top()), label)

        cv2.imshow("result", img)
        key = cv2.waitKey(-1) if image_dir else cv2.waitKey(30)

        if key == 27:  # ESC
            break


if __name__ == "__main__":
    main()
