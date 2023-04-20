# Fire Segmentation Project

This project uses deep learning to segment fire in open areas, such as woods or forests. The project is divided into two parts:

1. Data preparation
2. Training and evaluation

## Data preparation

The first step is to prepare the data. The data consists of a set of images and masks. The images are aerial images of fire, and the masks are binary masks that indicate the location of the fire. The data is split into a training set, a validation set, and a test set.
The dataset used in the experiment is available at the following [link](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs).

In particular, the user should download the images of fires (*9) Images for fire segmentation (Train/Val/Test) Images.zip*) and the related masks (*10) Masks annotation for fire segmentation (Train/Val/Test) Masks.zip*) zip files in the section *"Dataset Files"* and copy them in the folder `./data`.

## Training and evaluation

The next step is to train and evaluate the model. The model is trained on the training set, and its performance during training is evaluated on the validation set. The model is then evaluated on the test set.

The project uses a modified version of *DeepLabV3+* model, by implementing a *MobileNetV3* bottlenecks backbone and avoiding atrous convolution to enhance the segmentation speed. In addition, two extra shallow features are used in the decoder module to improve the segmentation accuracy.

The project uses the following metrics to evaluate the model:

* Mean Pixel Accuracy (MPA)
* Mean Intersection over Union (MIoU)
* Frames per second (FPS)

The results of the model are evaluated on the test set and compared to the outcomes of the original paper.

|          | Original Paper | Our results |
| :------- | :------------: | :---------: |
| MPA (%)  | 92.46          |  94.3       |
| MIoU (%) | 86.98          |  86.1       |
| FPS      | 59             |  55         |

## Instructions

To use the model, follow these steps:

1. Clone the repository.
2. Install the requirements.
3. Run the following command to train the model:

    `python script\train.py`

    The [execute first task script](src/execute_first_task.py), provides a command line script to execute the task of detecting the defects of a fruit. It firstly masks the fruit, then it looks for the defects. If the task is run in `verbose` mode, the visualization of the defect regions of the fruit is plotted.

    The script positional arguments are:
    * `fruit-image-path`: The path of the colour image of the fruit.
    * `fruit-nir-image-path`: The path of the Near Infra-Red image of the fruit.
    * `image-name` (optional): The name of the image.

    The following optional non positional arguments are present:
    * `--tweak-factor`, `-tf` (default=0.3): Tweak factor for obtaining the binary mask.
    * `--sigma`, `-s` (default=1): Sigma to apply to the Gaussian Blur operation before Canny's algorithm.
    * `--threshold-1`, `-t1` (default=60): First threshold that is used in Canny's algorithm hysteresis process.
    * `--threshold-2`, `-t2` (default=120): Second threshold that is used in Canny's algorithm hysteresis process.
    * `--no-verbose`, `-nv` (default=False); Skip the visualization of the results.


4. Run the following command to evaluate the results of the model:

    `python script\validate.py`

5. Run the following command to predict the fire segmentation on the test set:

   `python script\validate.py`
