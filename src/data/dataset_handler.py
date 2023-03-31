import re
from typing import Tuple
from zipfile import ZipFile
import cv2
from sklearn.model_selection import train_test_split
import numpy as np

def load_images_from_zip(zip_file_path: str, are_masks: bool) -> np.ndarray:
    """Load the images used for the segmentation task from a zip file.
    If the `are_mask` parameter is set to true, the images will be
    decoded in grayscale. Otherwise they will be decoded in BGR.

    Parameters
    ----------
    zip_file_path : str
        The path to the zip file containing the images.
    are_masks : bool
        If True, the images are assumed to be grayscale masks. If False,
        the images are assumed to be color images.

    Returns
    -------
    ndarray
        A numpy array containing the loaded images. If the images are
        grayscale masks, the shape of the array is (N, H, W) where N is
        the number of images, H is the height of each image, and W is
        the width of each image. If the images are color images, the shape
        of the array is (N, H, W, 3) where N is the number of images,
        H is the height of each image, W is the width of each image, and
        3 color channels are present (BGR).
    """
    images = []

    #TODO: remove
    i = 0

    with ZipFile(zip_file_path) as zf:
        # Get file names list and skip the first folder name.
        file_names = zf.namelist()[1:]
        # Sort the file names by their image number.
        file_names = sorted(
            file_names,
            key=lambda x: int(re.findall(r'[\d]+', x)[0]))

        for file_name in file_names:
            # Read the current file.
            data = zf.read(file_name)
            # Decode the file into a numpy array.
            if are_masks:
                img = cv2.imdecode(np.frombuffer(data, np.uint8),
                                   cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imdecode(np.frombuffer(data, np.uint8),
                                   cv2.IMREAD_COLOR)
            images.append(img)
            # TODO: remove
            i += 1
            if i == 500:
                break
    # Return the images as a numpy array.
    return np.array(images)

def resize_images(
    images: np.ndarray, resize_shape: Tuple[int, int]) -> np.ndarray:
    """Resize an array of images to a desired shape.

    Parameters
    ----------
    images : ndarray
        The array of images to reshape
    resize_shape : (int, int)
        The desired output shape of the images.

    Returns
    -------
    ndarray
        The reshaped input array of images.
    """
    images = [cv2.resize(img, resize_shape, interpolation=cv2.INTER_NEAREST)
              for img in images]

    return np.array(images)

def get_train_val_test_dataset_split(
    images: np.ndarray, masks: np.ndarray, test_size: float = .15,
    val_size: float = .15, seed: int = 42
    ) -> Tuple[Tuple[np.ndarray, np.ndarray],
               Tuple[np.ndarray, np.ndarray],
               Tuple[np.ndarray, np.ndarray]]:
    """Split the images and mask dataset into train, validation and test.

    Parameters
    ----------
    images : ndarray
        The images of the dataset.
    masks : ndarray
        The segmentation masks of the dataset.
    test_size : float, optional
        The test size ratio, by default 0.15.
    val_size : float, optional
        The validation size ratio, by default 0.15.
    seed : int, optional
        The seed to use for the split, by default 42.

    Returns
    -------
    (ndarray, ndarray)
        Tuple containing the input images and the segmentation masks
        of the train set.
    (ndarray, ndarray)
        Tuple containing the input images and the segmentation masks
        of the validation set.
    (ndarray, ndarray)
        Tuple containing the input images and the segmentation masks
        of the test set.
    """
    X_train, X_test, y_train, y_test =  train_test_split(
        images, masks, test_size=test_size, shuffle=True, random_state=seed)
    X_train, X_val, y_train, y_val =  train_test_split(
        X_train, y_train, test_size=val_size, shuffle=True, random_state=seed)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
