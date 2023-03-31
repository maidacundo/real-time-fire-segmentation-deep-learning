from time import time
from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from ..model.model import FireSegmentationModel
from ..training.training import resize_image_batch


def predict(
    model: FireSegmentationModel, dataloader: DataLoader, device: str,
    resize_evaluation_shape: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
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
            
            if resize_evaluation_shape is not None:
                # Resize the predictions
                y_pred = resize_image_batch(
                    y_pred, new_size=resize_evaluation_shape)
            results.extend(y_pred.cpu().numpy())

    # Remove unused tensors from gpu memory.
    torch.cuda.empty_cache()

    return np.array(results)