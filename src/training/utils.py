"""
Module providing the checkpoints monitor for the training
of the semantic segmentation module.
"""
import os
import torch
from torch import nn

class Checkpoint():
    """Class to handle the checkpoints of a model.

    Attributes
    ----------
    best_accuracy : float
        The best accuracy reached by the model.
    path : str
        The path where the checkpoints are saved.
    """
    def __init__(self, path: str, initial_accuracy: float = 0.) -> None:
        """Initialize the checkpoint instance.
        
        Parameters
        ----------
        path : str
            The checkpoint path.
        initial_accuracy : float, optional
            The initial accuracy value, by default 0.
        """
        # Set the best accuracy as the initial accuracy.
        self.best_accuracy = initial_accuracy

        # Set the checkpoint path.
        self.path = path
        # Create the checkpoint directory if it does not exist.
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def save_best(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                  new_accuracy: float) -> None:
        """
        Possibly save the best model weights and optimizer state 
        in the checkpoint file according to the new value of the metric.

        Parameters
        ----------
        model : Module
            The model which weights are saved.
        optimizer : Optimizer
            The optimizer which state is saved
        new_accuracy : float
            The new accuracy value which is compared to the best so far.
            The checkpoints are updated solely if the new accuracy is higher.
        """
        if new_accuracy > self.best_accuracy:
            # Set the new checkpoints.
            checkpoint = {}
            checkpoint['model_state_dict'] = model.state_dict()
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            checkpoint['best_accuracy'] = new_accuracy

            # Save the checkpoints.
            torch.save(checkpoint, self.path)
            # Update the best accuracy reached by the model.
            self.best_accuracy = new_accuracy

    def load_best_weights(self, model: nn.Module) -> None:
        """Load the best weights on a model.

        Parameters
        ----------
        model : Module
            The model for which the best weights are loaded.
        """
        # Load the checkpoints and set the model weights.
        checkpoint = torch.load(self.path)
        model.load_state_dict(checkpoint['model_state_dict'])
