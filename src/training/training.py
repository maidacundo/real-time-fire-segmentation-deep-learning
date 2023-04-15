from time import time
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Resize, InterpolationMode
import numpy as np

from .metrics import FocalLoss, MIoU, MPA
from .utils import Checkpoint
from ..model.model import FireSegmentationModel

def train(
    model: FireSegmentationModel, optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader, val_dataloader: DataLoader,
    epochs: int, validation_step: int, device: str,
    checkpoint: Optional[Checkpoint] = None, lr_schedulers: List[object] = [],
    reload_best_weights: bool = True) -> Dict[str, np.ndarray]:
    """
    Train a model for the semantic segmentation task of fires.

    Parameters
    ----------
    model : FireSegmentationModel
        The model to train.
    optimizer : Optimizer
        The optimizer used to train the model.
    train_dataloader : DataLoader
        The data loader used for training.
    val_dataloader : DataLoader
        The data loader used for validation.
    epochs : int
        The number of epochs to train the model for.
    validation_steps : int
        The number of steps to wait for applying validation.
    device : str
        The device to use for training.
    checkpoint : Checkpoint, optional
        The checkpoint used to save the best model, by default None.
    lr_schedulers : list of object, optional
        The learning rate schedulers for the optimizer.
        The first element of the list is the learning rate step
        scheduler, while the second is the learning rate scheduler
        that updates the learning rate on plateau, by default empty list.
    reload_best_weights : bool, optional
        Whether to reload the best weights after training, by default True.

    Returns
    -------
    { str: ndarray }
        A dictionary containing the training history.
    """
    # Initialize loss functions.
    criterion = FocalLoss()
    mpa_metric = MPA()
    miou_metric = MIoU()

    # Get the learning rate schedulers.
    step_lr_scheduler = lr_schedulers[0] if len(lr_schedulers) > 0 else None
    plateau_lr_scheduler = lr_schedulers[1] if len(lr_schedulers) > 1 else None

    # Initialize histories.
    metrics = ['train_loss', 'train_mpa', 'train_miou', 'val_loss', 'val_mpa',
               'val_miou']
    history = { m: [] for m in metrics }

    # Set model in training mode.
    model.train()

    # Iterate across the epochs.
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')

        # Remove unused tensors from gpu memory.
        torch.cuda.empty_cache()

        # Initialize running loss and accuracies.
        running_train_loss = 0.
        running_train_mpa = 0.
        running_train_miou = 0.

        start_time = time()

        for batch_idx, (x, y) in enumerate(train_dataloader):
            # Increment the number of batch steps.
            batch_steps = batch_idx + 1

            # Put the data to the desired device.
            x = x.to(device=device)
            y = y.to(device=device)

            # Compute model predictions.
            y_pred = model(x)

            # Compute the loss on the scaled results and ground truth.
            loss = criterion(y_pred, y)
            running_train_loss += loss.item()

            # Zero the gradients.
            optimizer.zero_grad()

            # Compute accuracies and update the running accuracies.
            mpa = mpa_metric(y_pred, y)
            miou = miou_metric(y_pred, y)

            running_train_mpa += mpa.item()
            running_train_miou += miou.item()

            # Use the loss function for backpropagation.
            loss.backward()

            # Update the weights.
            optimizer.step()

            # Increase the plateau learning rate scheduler step.
            plateau_lr_scheduler.step(running_train_loss / batch_steps)

            epoch_time = time() - start_time
            batch_time = epoch_time / batch_steps

            print(
                f'[{batch_steps}/{len(train_dataloader)}] -',
                f'{epoch_time:.0f}s {batch_time * 1e3:.0f}ms/step -',

                f'train {{ loss: {running_train_loss / batch_steps:.3g} -',
                f'MPA: {running_train_mpa * 100. / batch_steps:.3g}% -',
                f'MiOU: {running_train_miou * 100. / batch_steps:.3g}% }} -',

                f'lr: {optimizer.param_groups[0]["lr"]:.3g}',
                '             ' if batch_steps < len(train_dataloader) else '',
                end='\r')

            # Apply the validation step.
            if batch_steps % validation_step == 0:
                # Set the model in evaluation mode.
                model.eval()

                # Get the validation results.
                val_results = validate(model, val_dataloader, device)
                val_loss, val_mpa, val_miou, _ = val_results

                print(
                    '[VALIDATION] -',
                    f'{epoch_time:.0f}s -',

                    f'val: {{ loss: {val_loss:.3g} -',
                    f'MPA: {val_mpa * 100:.3g}% -',
                    f'MiOU: {val_miou * 100:.3g}% }} -',

                    f'lr: {optimizer.param_groups[0]["lr"]:.3g} -',
                    end='\r'
                )

                # Save the model checkpoints if demanded.
                if checkpoint is not None:
                    accuracy_sum = val_mpa + val_miou
                    checkpoint.save_best(model, optimizer, accuracy_sum)

                # Set model in training mode.
                model.train()

        # Set the model in evaluation mode.
        model.eval()

        # Get train results.
        train_loss = running_train_loss / len(train_dataloader)
        train_mpa = running_train_mpa / len(train_dataloader)
        train_miou = running_train_miou / len(train_dataloader)

        # Get validation results.
        val_results = validate(model, val_dataloader, device)
        val_loss, val_mpa, val_miou, _ = val_results

        # Update the training history.
        history['train_loss'].append(train_loss)
        history['train_mpa'].append(train_mpa)
        history['train_miou'].append(train_miou)

        history['val_loss'].append(val_loss)
        history['val_mpa'].append(val_mpa)
        history['val_miou'].append(val_miou)

        # Save the model checkpoints if demanded.
        if checkpoint is not None:
            accuracy_sum = val_mpa + val_miou
            checkpoint.save_best(model, optimizer, accuracy_sum)

        print(
            f'[{len(train_dataloader)}/{len(train_dataloader)}] -',
            f'{epoch_time:.0f}s -',

            f'train: {{ loss: {train_loss:.3g} -',
            f'MPA: {train_mpa * 100:.3g}% -',
            f'MiOU: {train_miou * 100:.3g}% }} -',

            f'val: {{ loss: {val_loss:.3g} -',
            f'MPA: {val_mpa * 100:.3g}% -',
            f'MiOU: {val_miou * 100:.3g}% }} -',

            f'lr: {optimizer.param_groups[0]["lr"]:.3g}',
            )

        # Increase the learning rate scheduler step.
        step_lr_scheduler.step()

        # Set model in training mode.
        model.train()

    # Load the best model weights if demanded.
    if checkpoint is not None and reload_best_weights:
        checkpoint.load_best_weights(model)

    # Set the model in evaluation mode.
    model.eval()

    # Remove unused tensors from gpu memory.
    torch.cuda.empty_cache()

    # Turn histories in numpy arrays.
    for k, v in history.items():
        history[k] = np.array(v)

    return history

def resize_image_batch(images: torch.FloatTensor,
                       new_size: Tuple[int, int]) -> torch.FloatTensor:
    resize = Resize(new_size, InterpolationMode.NEAREST)
    return resize(images)

def validate(
    model: FireSegmentationModel, val_dataloader: DataLoader, device: str,
    resize_evaluation_shape: Optional[Tuple[int, int]] = None
    ) -> Tuple[float, float, float, float]:
    """
    Validate the predictions of a model for the semantic segmentation
    task of fires. The loss, MPA (Mean Pixel Average), MIoU (Mean
    Intersection over Union) and FPS (Frames Per Second) are computed
    on the results with respect to the ground truth. 

    Parameters
    ----------
    model : FireSegmentationModel
        The model to use for validation.
    val_dataloader : DataLoader
        The data loader used for validation.
    device : str
        The device to use for validation.
    resize_evaluation_shape : (int, int), optional
        The target shape of the segmentation masks to use for
        evaluation. If not provided, the shape remains unchanged.
        By default None.

    Returns
    -------
    float
        The validation loss.
    float
        The validation MPA (Mean Pixel Average).
    float
        The validation MIoU (Mean Intersection over Union).
    float
        The validation FPS (Frames Per Second).
    """
    torch.cuda.empty_cache()
    model.eval()

    criterion = FocalLoss()
    mpa_metric = MPA()
    miou_metric = MIoU()

    running_val_loss = 0.
    running_val_mpa = 0.
    running_val_miou = 0.
    running_val_fps = 0.

    with torch.no_grad():
        
        for _, (x, y) in enumerate(val_dataloader):
            # Put the data to the desired device.
            x = x.to(device=device)
            y = y.to(device=device)

            # Compute the model predictions.
            torch.cuda.synchronize(device)
            start_time = time()
            y_pred = model(x)
            torch.cuda.synchronize(device)
            time_taken = time() - start_time

            if resize_evaluation_shape is not None:
                # Resize the predictions and the ground truth masks.
                y = resize_image_batch(y, new_size=resize_evaluation_shape)
                y_pred = resize_image_batch(
                    y_pred, new_size=resize_evaluation_shape)

            # Compute the loss on the results and ground truth.
            loss = criterion(y_pred, y)
            running_val_loss += loss.item()

            # Compute the accuracies.
            mpa = mpa_metric(y_pred, y)
            miou = miou_metric(y_pred, y)

            # Update the running metrics.
            running_val_mpa += mpa.item()
            running_val_miou += miou.item()
            running_val_fps += x.shape[0] / time_taken

    # Remove unused tensors from gpu memory.
    torch.cuda.empty_cache()

    # Get the average metrics.
    val_loss = running_val_loss / len(val_dataloader)
    val_mpa = running_val_mpa / len(val_dataloader)
    val_miou = running_val_miou / len(val_dataloader)
    fps = running_val_fps / len(val_dataloader)

    return val_loss, val_mpa, val_miou, fps
