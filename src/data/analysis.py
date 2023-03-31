from typing import Literal, Optional
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def plot_dataset_samples(images: np.ndarray, masks: np.ndarray,
                         num_samples: int = 3, 
                         title: Optional[str] = None) -> None:
    """
    Plot a desired number of dataset samples.
    On the first row the original images are plotted, on the second
    row their masks are shown and on the third one the ROI of
    the fire is highlighted over the image.
    The samples are fetched in a way that their indices are equidistant
    from one to the other in the original array.

    Parameters
    ----------
    images : ndarray
        Numpy array of input images.
    masks : np.ndarray
        Numpy array of the segmentation masks of the given images.
    num_samples : int, optional
        The number of samples to plot, by default 3.
    """
    # Get equidistant sample indices for the images in the dataset.
    sample_indices = np.linspace(0, len(images) - 1, num=num_samples,
                                 dtype=int)
    _, axes = plt.subplots(3, num_samples, figsize=(15, 8))

    for i, idx in enumerate(sample_indices):
        # Plot color image.
        ax = axes[0, i]
        ax.imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB))
        ax.axis('off')

        # Plot mask.
        ax = axes[1, i]
        ax.imshow(masks[idx], cmap='gray', vmin=0., vmax=1.)
        ax.axis('off')
        legend_elements = [
            Patch(facecolor='w', edgecolor='black',label='Fire mask')]
        ax.legend(handles=legend_elements, loc='upper left')

        # Plot highlighted mask over the color image.
        ax = axes[2, i]
        highlighted_roi = _get_highlighted_roi_by_mask(
            images[idx], masks[idx], highlight_channel='red')
        ax.imshow(cv2.cvtColor(highlighted_roi, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        legend_elements = [
            Patch(facecolor='r', edgecolor='black', label='Fire ROI')]
        ax.legend(handles=legend_elements, loc='upper left')
    if title is None:
        title = f'Representation of {num_samples} images from the dataset '
        'along with their fire mask and the highlighted segmentation'
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def _get_highlighted_roi_by_mask(
    image: np.ndarray, mask: np.ndarray,
    highlight_channel: Literal['blue', 'green', 'red'] = 'green'
    ) -> np.ndarray:
    """
    Function that highlights a Region of Interest provided by a mask over an
    image with a given BGR colour.

    Parameters
    ----------
    image: ndarray
        Image on which the mask is highlighted.
    mask: ndarray
        Mask illustrating the Region of Interest to highlight over the
        image.
    highlight_channel: str, optional
        Colour of the highlighted mask: 'blue'; 'green' or 'red', by 
        default: 'green'.

    Returns
    -------
    highlighted_roi: ndarray
        Highlighted Region of Interest over the input image.
    """
    channel_map = { 'blue': 0, 'green': 1, 'red': 2 }

    # Turn the mask into a BGR image.
    mask = mask.astype(np.uint8)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Force the bits of every channel except the selected one at 0.
    for c, v in channel_map.items():
        if c != highlight_channel:
            mask[..., v] = 0

    mask[mask == 1.] = 255
    # Highlight the unmasked ROI.
    return cv2.addWeighted(mask, 0.9, image, 1, 0)
