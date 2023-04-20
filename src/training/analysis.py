"""
Module containing functions for analyzing the training history of the model.
"""
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np

def _plot_subplot(
    index: int, train_history: np.ndarray, val_history: np.ndarray,
    title: str, metric_name: str) -> None:
    """Plot a training history subplot for a specific metric.

    Parameters
    ----------
    index : int
        The subplot index.
    train_history : ndarray
        The training history of a specific metric.
    val_history : ndarray
        The validation history of a specific metric.
    title : str
        The title of the subplot.
    metric_name : str
        The name of the considered metric.
    """
    # Plot the subplot at the given index.
    plt.subplot(*index)
    # Set the title.
    plt.title(title)

    # Plot the training and validation history.
    plt.plot(train_history, label='train')
    plt.plot(val_history, label='validation')

    # Set the x and y labels.
    plt.xlabel('epochs')
    plt.ylabel(metric_name)

    # Plot the legend.
    plt.legend()

def plot_training_history(history: Dict[str, np.ndarray]) -> None:
    """Plot the training history of the model.

    Parameters
    ----------
    history : { str: ndarray }
        A dictionary containing the training history values, including:
        * Train loss history.
        * Validation loss history.
        * Train MIoU (Mean Intersection over Union) history.
        * Validation MIoU history.
        * Train MPA (Mean Pixel Average) history.
        * Validation MPA history.
    """
    plt.figure(figsize=(15, 10))

    # Plot the training history subplots.
    _plot_subplot((2, 1, 1), history['train_loss'], history['val_loss'],
                  'Train and validation loss history', 'loss')
    _plot_subplot((2, 2, 3), history['train_miou'], history['val_miou'],
                  'Train and validation MIoU history', 'MIoU')
    _plot_subplot((2, 2, 4), history['train_mpa'],
                  history['val_mpa'],
                  'Train and validation MPA history', 'MPA')

    # Set the title.
    plt.suptitle('Training and validation history', size=16)

    # Configure the layout and plot.
    plt.tight_layout()
    plt.show()
