# Convolutional Neural Network for Image Classification

Welcome to the repository for our image classification project using Convolutional Neural Networks (CNN). The purpose of this project is to classify medical scans into different categories, like "normal" and "pneumonia".

## Overview

This project involves the construction of a CNN with an architecture formed by stacking processing layers:

1. Convolutional Layers (CONV)
2. Pooling Layers (POOL)
3. Batch-normalization Layers
4. Fully Connected Layers (FC)

We use 5 convolutional blocks, each consisting of a convolutional layer, pooling layer, and a batch-normalization layer.

## Architecture Details

- **Convolutional Layer (CONV)**: Processes data from a receptive field. This layer has hyperparameters like depth (number of neurons associated with the same receptive field), stride, and zero-padding.
  
- **Pooling Layer (POOL)**: Compresses the information by reducing the size of the intermediate image, commonly using the Max-Pool 2x2 method.

- **Batch-normalization Layer**: Normalizes the inputs of each layer, making the network faster and more stable.

- **Fully Connected Layer (FC)**: Connects every neuron from one layer to every neuron of another layer. It is always the last layer in our network, giving the final probability vector for the classes.

In addition, we use the dropout method in each block to reduce overfitting.

## Activation Functions

The activation functions used in this project include:

- Tangent Hyperbolic: f(x) = tanh(x)
- Absolute Tangent Hyperbolic: f(x) = |tanh(x)|
- Sigmoid: f(x) = (1 + e^(-x))^(-1)
- Rectified Linear Unit (ReLU): f(x) = max(0,x)

## Getting Started

1. **Prerequisites**: Ensure you have [Python](https://www.python.org/downloads/) and the required libraries installed.

2. **Installation**: Clone this repository and navigate to the project folder. Run `pip install -r requirements.txt` to install the necessary packages.

3. **Training**: To train the model, execute `python train_model.py`.

4. **Testing**: After training, you can test the model using `python test_model.py`.

## Further Reading

1. Dropout method: [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html)

2. Speeding up CNNs: [ImageNet Classification with Deep Convolutional Neural Networks](https://www.cs.toronto.edu/~kriz/imagenet_classification_with_deep_convolutional.pdf)

## Contributing

Contributions, issues, and feature requests are welcome!

## License

This project is [MIT](https://choosealicense.com/licenses/mit/) licensed.

---

You can further customize this README based on the specific features, requirements, and intricacies of your project.
