import sys
sys.path.append('.')

import argparse
import os
import numpy as np
from torch import cuda

from src.data.dataset_handler import (
    load_images_from_zip, get_train_val_test_dataset_split)
from src.data.dataloaders import get_dataloader
from src.model.model import FireSegmentationModel
from src.training.training import validate
from src.training.utils import Checkpoint


def main():
    # Set the argument parser.
    parser = argparse.ArgumentParser(
        description='Script for validating the results of the fire detection '
        'segmentation model.')

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
        help='The path of the file where the model checkpoints are loaded.',
        nargs='?', required=False)

    parser.add_argument(
        '--train-mean-std-file-path','-ms', metavar='Mean and std file path',
        type=str, default=os.path.join('model', 'mean-std.npy'),
        help='The file path where the train mean and standard deviation are '
        'loaded', nargs='?', required=False)

    parser.add_argument(
        '--seed', '-s', type=int, default=42, nargs='?',
        help='The seed used for reproducibility.', required=False)

    parser.add_argument(
        '--device', '-d', type=str, default=None, nargs='?',
        help='The device to use for training. If not provided, it is set '
        'automatically.', required=False)

    parser.add_argument(
        '--eval-batch-size', '-eb', type=int, default=2, nargs='?',
        help='The batch size used for evaluation.', required=False)

    # Get the arguments.
    arguments = parser.parse_args()

    images_zip_path = arguments.images_zip_path
    masks_zip_path = arguments.masks_zip_path
    chekpoint_file_path = arguments.checkpoint_file_path
    train_mean_std_file_path = arguments.train_mean_std_file_path
    seed = arguments.seed
    device = arguments.device
    eval_batch_size = arguments.eval_batch_size

    # Set the original shape.
    ORIGINAL_SHAPE = (3840, 2160)
    # Set the resize shape.
    RESIZE_SHAPE = (512, 512)
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
    _, (X_val, y_val), (X_test, y_test) = get_train_val_test_dataset_split(
        images, masks, seed=seed)

    # Set the model.
    model = FireSegmentationModel(RESIZE_SHAPE, device=device)
    model.eval()
    # Load the best weights of the model.
    checkpoint = Checkpoint(chekpoint_file_path)
    checkpoint.load_best_weights(model)

    # Load the mean and std of the training set for applying normalization.
    train_mean, train_std = np.load(train_mean_std_file_path)

    # Set the train and validation dataloaders.
    print('Building the dataloaders...')

    val_loader = get_dataloader(
        X_val, y_val, train_mean, train_std, batch_size=eval_batch_size,
        shuffle=False, apply_augmentation=False)

    test_loader = get_dataloader(
        X_test, y_test, train_mean, train_std, batch_size=eval_batch_size,
        shuffle=False, apply_augmentation=False)

    print('Start evaluation on the validation set...')

    # Get the validation evaluation results.
    val_loss, val_mpa, val_miou, val_fps = validate(
        model, val_loader, device, resize_evaluation_shape=ORIGINAL_SHAPE)

    print('Validation loss:', f'{val_loss:.3g}')
    print('Validation MPA:', f'{val_mpa * 100:.3g}')
    print('Validation MIoU:', f'{val_miou * 100:.3g}')
    print('Validation FPS:', f'{val_fps:.3g}')

    print('Start evaluation on the validation set...')

    # Get the test evaluation results.
    val_loss, val_mpa, val_miou, val_fps = validate(
        model, test_loader, device, resize_evaluation_shape=ORIGINAL_SHAPE)

    print('Test loss:', f'{val_loss:.3g}')
    print('Test MPA:', f'{val_mpa * 100:.3g}')
    print('Test MIoU:', f'{val_miou * 100:.3g}')
    print('Test FPS:', f'{val_fps:.3g}')

if __name__ == '__main__':
    main()
