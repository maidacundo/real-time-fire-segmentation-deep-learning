"""Module for predicting the segmentation masks of images."""
from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from ..model.model import FireSegmentationModel
from ..training.training import resize_image_batch


def predict(
    model: FireSegmentationModel, dataloader: DataLoader, device: str
    ) -> np.ndarray:
    """
    Get the predicted segmentation mask from the data of a
    dataloader.

    Parameters
    ----------
    model : FireSegmentationModel
        The model to use for the predictions.
    dataloader : DataLoader
        The data loader which data should be predicted.
    device : str
        The device to use for predicting.

    Returns
    -------
    ndarray
        The predicted and eventually reshaped sematic segmentation
        masks of the images of the dataloader.
    """
    # Remove unused tensors from the GPU.
    torch.cuda.empty_cache()

    # Set the model in evaluation mode.
    model.eval()

    # Set the results array.
    results = []

    with torch.no_grad():
        for _, (x, _) in enumerate(dataloader):
            # Put the data to the desired device.
            x = x.to(device=device)

            # Compute the model predictions.
            y_pred = model(x)
            y_pred = y_pred.softmax(-3).argmax(-3)

            results.extend(y_pred.cpu().numpy())

    # Remove unused tensors from gpu memory.
    torch.cuda.empty_cache()

    return np.array(results)