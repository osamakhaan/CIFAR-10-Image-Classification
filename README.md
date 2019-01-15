# CIFAR-10-Image-Classification
Image classification on the CIFAR 10 Dataset using Support Vector Machines (SVMs), Fully Connected Neural Networks and Convolutional Neural Networks (CNNs). The files are organized as follows:

**SVMs_Part1** -- Image Classification on the CIFAR-10 Dataset using Support Vector Machines. Different types of kernels are used including Linear Kernel, Polynomial Kernel and the Radial Basis Function (RBF) Kernel.

**SVMs_Part2_PCA** -- Image Classification on the CIFAR-10 Dataset using Support Vector Machines. Principal Component Analysis (PCA) is used for dimensionality reduction. The number of dimensions are chosen based on the cumulative explained variance, as shown below:

| Number of Components  | Percentage variance explained |
| ------------- | ------------- |
| 150  | 93.1  |
| 500  | 98.5  |

Test results for the two transformations are shown below:

| Number of Components  | Test Set Accuracy |
| ------------- | ------------- |
| 150  | 40.2  |
| 500  | 39.6  |

**CNNs_Part1** -- Three different types of models are used in this case.

1. A Fully Connected Neural Network (3 Layers).
2. A Fully Convolutional Neural Network (4 Layers).
3. A Hybrid Model containing both convolutional and fully-connected layers.

The results are tabulated below:

| Model  | Test Set Accuracy |
| ------------- | ------------- |
| Fully Connected DNN  | 54  |
| Fully Convolutional NN  | 71 |
| Hybrid Model  | 66 |


**CNNs_Part2** -- A Python File for training the convolutional neural network on the GPU.


See "Final_Report.pdf" for an analysis of the efficacy of the various algorithms on the CIFAR-10 Image Classification task.


