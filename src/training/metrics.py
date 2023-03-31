"""
Module providing the necessary metrics to train and evaluate the
model for the semantic segmentation task.
"""
import torch
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for the semantic segmentation tasks.

    Attributes
    ----------
    alpha : float
        Balancing parameter for class weights.
    gamma : float
        Focusing parameter to adjust the loss contribution of
        well-classified and misclassified examples.
        
    Methods
    -------
    forward(inputs: FloatTensor, targets: FloatTensor) -> FloatTensor:
        Apply the Focal Loss between the model predictions and
        the ground truth target tensors.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.) -> None:
        """Initialize the Focal Loss module.

        Parameters
        ----------
        alpha : float, optional
            Balancing parameter for class weights, by default 0.25.
        gamma : float, optional
            Focusing parameter to adjust the loss contribution of
            well-classified and misclassified examples. By default 2.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.FloatTensor, targets: torch.FloatTensor
                ) -> torch.FloatTensor:
        """Compute the forward pass of the Focal Loss.

        Parameters
        ----------
        inputs : FloatTensor
            Tensor of the model predictions.
        targets : FloatTensor
            Tensor of the ground truth values.

        Returns
        -------
        FloatTensor
            Scalar value of the Focal Loss.
        """
        # Get the cross entropy without reduction.
        cross_entropy_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none')

        # Compute the pt value and the focal loss.
        pt = torch.exp(-cross_entropy_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * cross_entropy_loss
        return focal_loss.mean()

class MIoU(nn.Module):
    """
    Mean Intersection over Union (MIoU) implementation for the
    semantic segmentation tasks.

    Attributes
    ----------
    smooth : float
        Smoothing value to avoid division by zero.

    Methods
    -------
    forward(inputs: FloatTensor, targets: FloatTensor) -> FloatTensor:
        Apply the MIoU between the model predictions and
        the ground truth target tensors.
    """
    def __init__(self, smooth: float = 1e-6) -> None:
        """Initialize the MIoU module.

        Parameters
        ----------
        smooth : float, optional
            Smoothing value to avoid division by zero,
            by default 1e-6.
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.FloatTensor, targets: torch.FloatTensor
                ) -> torch.FloatTensor:
        """Compute the forward pass of the MIoU.

        Parameters
        ----------
        inputs : FloatTensor
            Tensor of the model predictions.
        targets : FloatTensor
            Tensor of the ground truth values.

        Returns
        -------
        FloatTensor
            Scalar value of the MIoU.
        """
        with torch.no_grad():
            # Get the image predictions through softmax and argmax
            # (0: background, 1: foreground).
            inputs = inputs.softmax(-3).argmax(-3, keepdim=True)

            # Concatenate the background and foreground predictions
            # channel-wise.
            bg_predictions = ~inputs.bool()
            fg_predictions = inputs.bool()
            inputs = torch.cat([bg_predictions, fg_predictions], dim=-3)

            # Turn the ground truth to a boolean tensor.
            targets = targets.bool()

            # Get the intersection.
            intersection = (inputs & targets).sum((-2, -1))
            # Get the union.
            union = (inputs | targets).sum((-2, -1))

            # Compute the MIoU.
            iou = (intersection + self.smooth) / (union + self.smooth)
            return torch.mean(iou)

class MPA(nn.Module):
    """
    Mean Pixel Average (MPA) implementation for the semantic
    segmentation tasks.

    Attributes
    ----------
    smooth : float
        Smoothing value to avoid division by zero.

    Methods
    -------
    forward(inputs: FloatTensor, targets: FloatTensor) -> FloatTensor:
        Apply the MPA between the model predictions and
        the ground truth target tensors.
    """
    def __init__(self, smooth: float = 1e-6) -> None:
        """Initialize the MPA module.

        Parameters
        ----------
        smooth : float, optional
            Smoothing value to avoid division by zero,
            by default 1e-6.
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.FloatTensor, targets: torch.FloatTensor
                ) -> torch.FloatTensor:
        """Compute the forward pass of the MPA.

        Parameters
        ----------
        inputs : FloatTensor
            Tensor of the model predictions.
        targets : FloatTensor
            Tensor of the ground truth values.

        Returns
        -------
        FloatTensor
            Scalar value of the MPA.
        """
        with torch.no_grad():
            # Get the image predictions through softmax and argmax
            # (0: background, 1: foreground).
            inputs = inputs.softmax(-3).argmax(-3, keepdim=True)

            # Concatenate the background and foreground predictions
            # channel-wise.
            bg_predictions = ~inputs.bool()
            fg_predictions = inputs.bool()
            inputs = torch.cat([bg_predictions, fg_predictions], dim=-3)

            # Turn the ground truth to a boolean tensor.
            targets = targets.bool()

            # Get the intersection.
            intersection = (inputs & targets).sum((-2, -1))
            # Get the total number of elements per image (pixel count).
            total_pixels = targets.sum((-2, -1))

            # Compute the MPA.
            mpa = (intersection + self.smooth) / (total_pixels + self.smooth)
            return torch.mean(mpa)
