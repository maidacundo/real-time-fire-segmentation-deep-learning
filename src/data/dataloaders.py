"""Module containing the dataloaders for the fire segmentation task."""
from typing import Tuple
import torch
from torch.utils.data.dataloader import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np


class FireDetectionDataset(Dataset):
    """Class implementing a dataset for the fire segmentation task.

    Attibutes
    ---------
    X : ndarray
        The input images of the fire segmentation dataset.
    y : ndarray
        The segmentation masks of the input images of the dataset.
    len : int
        The length of the dataset.
    transform_image_and_mask : Compose
        Transform functions composition to apply on both the images
        and masks.
    transform_image : ColorJitter
        Color Jitter transformation to apply solely on the images.
    to_tensor : ToTensor
        Function to transform the images and masks to tensors.
    normalize : Normalize
        Normalization function applied solely on the images.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, train_mean: np.ndarray,
                 train_std: np.ndarray, apply_augmentation: bool) -> None:
        """Initialize the fire segmentation dataset.

        Parameters
        ----------
        X : ndarray
            The input images of the fire segmentation dataset.
        y : ndarray
            The segmentation masks of the input images of the dataset.
        train_mean : ndarray
            The mean value per channel of the training dataset. It is
            used as an estimation to apply the standard scaling of
            the images for increasing the performances of the model.
        train_std : ndarray
            The standard deviation per channel of the training dataset. It is
            used as an estimation to apply the standard scaling of
            the images for increasing the performances of the model.
        apply_augmentation : bool
            Whether to apply image augmentation while fetching the images
            or not.
        """
        super().__init__()
        assert X.shape[:-1] == y.shape, 'The image does not have the same ' +\
            'shape as the mask.'
        self.X = X
        self.y = y
        self.len = X.shape[0]
        # Set the image and mask transformations if demanded.
        self.transform_image_and_mask = T.Compose([
            T.RandomPerspective(distortion_scale=.3),
            T.RandomHorizontalFlip(),
            T.RandomAffine(degrees=(-45, 45), translate=(0.1, 0.1),
                           scale=(0.5, 1.5))
            ]) if apply_augmentation else None
        # Set the image transformations if demanded.
        self.transform_image = T.ColorJitter(brightness=.2, hue=.05) \
            if apply_augmentation else None
        # Set the tensor transformation.
        self.to_tensor = T.ToTensor()
        # Set the normalization transformation.
        self.normalize = T.Normalize(mean=train_mean, std=train_std)

    def __getitem__(self, index: int
                    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Get the image and the respective segmentation mask at the
        provided index.
        Data augmentation is applied if demanded.
        The segmentation mask is returned as the background mask
        and the foreground mask defining the fire.

        Parameters
        ----------
        index : int
            The index where the data is fetched.

        Returns
        -------
        FloatTensor
            The image at the given index.
        FloatTensor
            The respective segmentation mask at the given index.
        """
        # Get the data at the given index.
        X, y = self.X[index], self.y[index]

        # Turn the data into a tensor image.
        X = self.to_tensor(X)
        # Turn the mask as a 0/255 value vector for transformation.
        y[y == 1] = 255
        # Turn the mask into a tensor image.
        y = self.to_tensor(y)

        # Apply augmentations to both the image and the mask if demanded.
        if self.transform_image_and_mask is not None:
            # Increase to 3 the number of channels of the mask.
            y = y.expand_as(X)
            # Concatenate the image and the mask.
            X_y = torch.cat([X.unsqueeze(0), y.unsqueeze(0)], dim=0)
            # Apply the same transformations to the image and the mask.
            X_y = self.transform_image_and_mask(X_y)
            # Split the image and the mask.
            X, y = X_y[0], X_y[1]
            # Turn the mask into a 1 channel image.
            y = y[:1]
        # Apply augmentations to the image if demanded.
        if self.transform_image is not None:
            X = self.transform_image(X)
        # Apply standard scaling to the image.
        X = self.normalize(X)

        # Create a background mask that is the opposite of the original
        # fire mask.
        y_b = (~y.bool()).float()
        y = y.bool().float()
        # Create a 2 channel mask containing the semantic segmentation
        # of the background and the foreground.
        y = torch.cat([y_b, y], dim=0)
        return X, y

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.
        """
        return self.len

def get_dataloader(
    X: np.ndarray, y: np.ndarray, train_mean: np.ndarray, 
    train_std: np.ndarray, batch_size: int, shuffle: bool,
    apply_augmentation: bool) -> DataLoader:
    """Get a dataloader for the fire sematic segmentation task.

    Parameters
    ----------
    X : ndarray
        The input images of the fire segmentation dataset.
    y : ndarray
        The segmentation masks of the input images of the dataset.
    train_mean : ndarray
        The mean value per channel of the training dataset. It is
        used as an estimation to apply the standard scaling of
        the images for increasing the performances of the model.
    train_std : ndarray
        The standard deviation per channel of the training dataset. It is
        used as an estimation to apply the standard scaling of
        the images for increasing the performances of the model.
    batch_size : int
        The batch size to use for fetching the images.
    shuffle : bool
        Whether to shuffle the dataset or not before fetching.
    apply_augmentation : bool
        Whether to apply image augmentation while fetching the images
        or not.

    Returns
    -------
    DataLoader
        The dataloader.
    """
    return DataLoader(
        FireDetectionDataset(X, y, train_mean, train_std,
                             apply_augmentation=apply_augmentation),
        batch_size=batch_size, shuffle=shuffle, drop_last=True)
