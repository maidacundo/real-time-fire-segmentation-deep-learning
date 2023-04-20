import sys
sys.path.append('.')

import argparse
import os
import numpy as np
from torch import cuda
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from src.data.dataset_handler import (
    load_images_from_zip, get_train_val_test_dataset_split)
from src.data.dataloaders import get_dataloader
from src.model.model import FireSegmentationModel
from src.training.training import train
from src.training.lion import Lion
from src.training.utils import Checkpoint


def main():
    # Set the argument parser.
    parser = argparse.ArgumentParser(
        description='Script for training the fire detection segmentation '
        'model.')

    # Set the script arguments.
    parser.add_argument(
        '--images-zip-path', '-imgs', metavar='Images zip path', type=str,
        help='The path of the aerial images of the woodland fires zip file.',
        default=os.path.join('data', 'Images.zip'), nargs='?', required=False)

    parser.add_argument(
        '--masks-zip-path','-msks', metavar='Masks zip path', type=str,
        help='The path of the woodland fires segmentation masks zip file.',
        default=os.path.join('data', 'Masks.zip'), nargs='?', required=False)

    parser.add_argument(
        '--checkpoint-file-path','-ckpt', metavar='Checkpoint file path',
        type=str, default=os.path.join('model', 'checkpoints.pth'),
        help='The path of the file where the model checkpoints are saved.',
        nargs='?', required=False)

    parser.add_argument(
        '--train-mean-std-file-path','-ms', metavar='Mean and std file path',
        type=str, default=os.path.join('model', 'mean-std.npy'),
        help='The file path where the train mean and standard deviation are '
        'saved', nargs='?', required=False)

    parser.add_argument(
        '--seed', '-s', type=int, default=42, nargs='?',
        help='The seed used for reproducibility.', required=False)

    parser.add_argument(
        '--device', '-d', type=str, default=None, nargs='?',
        help='The device to use for training. If not provided, it is set '
        'automatically.', required=False)

    parser.add_argument(
        '--train-batch-size', '-tb', type=int, default=2, nargs='?',
        help='The batch size used for training.', required=False)

    parser.add_argument(
        '--eval-batch-size', '-eb', type=int, default=2, nargs='?',
        help='The batch size used for evaluation.', required=False)

    parser.add_argument(
        '--epochs', '-e', type=int, default=30, nargs='?',
        help='The number of epochs to train the model.', required=False)

    # Get the arguments.
    arguments = parser.parse_args()

    images_zip_path = arguments.images_zip_path
    masks_zip_path = arguments.masks_zip_path
    chekpoint_file_path = arguments.checkpoint_file_path
    train_mean_std_file_path = arguments.train_mean_std_file_path
    seed = arguments.seed
    device = arguments.device
    train_batch_size = arguments.train_batch_size
    eval_batch_size = arguments.eval_batch_size
    epochs = arguments.epochs

    # Set the resize shape.
    RESIZE_SHAPE = (512, 512)
    # Set the validation step.
    VAL_STEP = 200
    # Set the device.
    if device is None:
        device = 'cuda' if cuda.is_available() else 'cpu'

    # Get the images and masks.
    print('Loading the images...')
    images = load_images_from_zip(images_zip_path,
                              are_masks=False,
                              resize_shape=RESIZE_SHAPE)

    print('Loading the masks...')
    masks = load_images_from_zip(masks_zip_path,
                             are_masks=True,
                             resize_shape=RESIZE_SHAPE)

    # Split the dataset into train and validation sets.
    print('Splitting the dataset...')
    (X_train, y_train), (X_val, y_val), _ = get_train_val_test_dataset_split(
        images, masks, seed=seed)

    # Set the model.
    model = FireSegmentationModel(RESIZE_SHAPE, device=device)

    # Compute the mean and std of the training set for applying normalization.
    print('Computing the mean and std of the training set...')
    train_mean = np.mean(X_train, axis=(-4, -3, -2))
    train_std = np.std(X_train, axis=(-4, -3, -2))

    # Save the mean and std of the training set.
    train_mean_std_file_dir = os.path.dirname(train_mean_std_file_path)
    os.makedirs(train_mean_std_file_dir, exist_ok=True)
    np.save(train_mean_std_file_path, (train_mean, train_std))

    # Set the train and validation dataloaders.
    print('Building the dataloaders...')
    train_loader = get_dataloader(
        X_train, y_train, train_mean, train_std, batch_size=train_batch_size,
        shuffle=True, apply_augmentation=True)

    val_loader = get_dataloader(
        X_val, y_val, train_mean, train_std, batch_size=eval_batch_size,
        shuffle=False, apply_augmentation=False)

    # Set the optimizer.
    optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Set the Learning Rate Schedulers.
    step_lr_scheduler = StepLR(optimizer, gamma=.94, step_size=1)
    plateau_lr_scheduler = ReduceLROnPlateau(
        optimizer, factor=.98, patience=300, threshold=1e-6)
    lr_schedulers=[step_lr_scheduler, plateau_lr_scheduler]

    checkpoint = Checkpoint(chekpoint_file_path)

    print('Start training...')
    _ = train(model, optimizer, train_loader, val_loader, epochs, VAL_STEP,
                device, checkpoint, lr_schedulers, reload_best_weights=True)

if __name__ == '__main__':
    main()
