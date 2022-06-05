# Identifying-Fruits-and-Vegetables-in-Images
Assignment for the Masters Course Applications of Image and Video Processing

Advancements in deep learning and computer vision continue to push the boundaries of what is possible. Sometimes however it is important to look back at the work done in the past to understand the technology of today. In this project, smaller convolutional neural networks are explored in an attempt to discover the limits and capabilities of networks that can be run with lower hardware requirements and much lower training times. The goal of this project is to classify images of fruits and vegetables from the fruits360 dataset. Once this is achieved, the images will be obscured so as to test the ability of the network to adapt to different images that have information missing. These new images will also be used to train networks to test the effectiveness of an obscured dataset in classifying the original images, along with images where different sections have been eliminated. This leads to the following research question: Will a small, efficient neural network be able to classify obscured images, and will the obscured images train a network more effectively for the same task?

Dataset: https://www.kaggle.com/datasets/moltean/fruits

Packages required:
OpenCV
Numpy
Pytorch

All files include original paths and will need to be adjusted.

Can select the desired dataset for training in the two network python files
Can select the desired model and test dataset in the two test python files

imageSegmentation.py not used in final results
