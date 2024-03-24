## SRGAN - Image Super-Resolution with Generative Adversarial Networks
This repository implements the SRGAN architecture from the research paper "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network: https://arxiv.org/abs/1609.04802" by Christian Ledig et al. It utilizes Convolutional Neural Networks (CNNs) to upscale low-resolution images (32x32 pixels) to a higher resolution (128x128 pixels).  This implementation incorporates Keras Tuner for automated hyperparameter tuning.

### Project Structure:

classes.py: Defines the architecture of the generator and discriminator models, along with a customized loss function using VGG19 for perceptual similarity.
main.ipynb: Contains Jupyter Notebook code for initialization, data preprocessing, training loop, and visualization of results.
dataset_faces.rar: Compressed archive containing the training dataset. This dataset includes human faces downsampled to both 128x128 (high resolution) and 32x32 (low resolution) for training purposes.
Projects_checkpoints: Folder containing multiple trained models (generator and discriminator) saved in the H5 format. These can be loaded and used for image super-resolution.

### Usage:

Download the repository and extract the dataset_faces.rar file.
Install required libraries: Ensure you have Python, TensorFlow, Keras, Keras Tuner, and other relevant libraries installed according to the paper's requirements.
Run the training script: Open main.ipynb in a Jupyter Notebook environment and execute the cells to initialize, preprocess data, train the models, and visualize results.

### Pre-trained Models:

The Projects_checkpoints folder provides pre-trained models you can load and use for super-resolution tasks. Refer to the comments within main.ipynb for details on loading these models.

Disclaimer: The provided dataset (dataset_faces.rar) might contain copyright-protected images.  Please ensure you have the necessary rights to use the dataset for your purposes.
