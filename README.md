# Image Classification Model

This project is an image classification model using Convolutional Neural Networks (CNN) to classify images into various categories such as buildings, forests, glaciers, mountains, seas, and streets.

## Project Overview

This project uses a CNN model implemented in TensorFlow/Keras to classify images into six different categories. The dataset consists of training, testing, and prediction images structured into folders. The model is trained on the training dataset and evaluated on the testing dataset.

## Installation

To set up the project, clone this repository and install the required packages using `pip`:
git clone <repository-url>
cd <repository-directory>
pip install -r requirements.txt
Replace <repository-url> with the actual URL of your repository and <repository-directory> with the name of the cloned directory.

Data Structure
The data is organized into the following structure:

Copy code
project-directory/
├── seg_train/
│   ├── buildings/
│   ├── forest/
│   ├── glacier/
│   ├── mountain/
│   ├── sea/
│   └── street/
├── seg_test/
│   ├── buildings/
│   ├── forest/
│   ├── glacier/
│   ├── mountain/
│   ├── sea/
│   └── street/
└── seg_pred/
    └── images_to_predict/
seg_train/: Contains subfolders for each category of images for training.
seg_test/: Contains subfolders for each category of images for testing.
seg_pred/: Contains images for prediction.
Usage
Place your images in the appropriate folders as described in the data structure section.
Run the Python script containing the model implementation:
bash
Copy code
python your_script.py
Replace your_script.py with the name of your script.

The model will train for 40 epochs, evaluate on the test dataset, and save the trained model as image_classification_model.h5.
Model Architecture
The CNN model architecture is as follows:

Conv2D Layers: Three convolutional layers with ReLU activation.
MaxPooling Layers: Two max pooling layers.
Flatten Layer: Flattens the 3D output to 1D.
Dense Layers: Three fully connected layers with ReLU activation.
Dropout Layer: A dropout layer with a rate of 0.5 to prevent overfitting.
Output Layer: A softmax output layer for classification into 6 categories.
Results
The model's performance can be evaluated using the test dataset. After training, the model will predict the classes for images in the prediction folder, and the results will be displayed visually.
