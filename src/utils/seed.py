"""Module providing a functions to set the seed."""
import random
import cv2
import torch
import numpy as np


def set_random_seed(random_seed: int = 42) -> None:
    """Set the random seed for reproducibility. The seed is set for the 
    random library, the numpy library and the OpenCV ansìd pytorch 
    libraries.
    
    Parameters
    ----------
    random_seed : int, optional
        The random seed to use for reproducibility, by default 42.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    cv2.setRNGSeed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True