"""
An experiment for training a neural network on the MNIST dataset.

This script defines and trains a simple feed-forward neural network
to classify handwritten digits from the MNIST dataset.

The script performs the following main operations:
    1. Loads the MNIST dataset using `data_utils.load_mnist`.
    2. Defines a `NeuralNetwork` with the following architecture:
        - A `Flatten` layer to convert the 28x28 input images into a 1D array.
        - Two hidden `Dense` layers, each with 128 neurons and ReLU activation.
        - A `Dense` output layer with 10 neurons (one for each digit) and
          a Softmax activation function.
    3. Compiles the network with the Adam optimizer (learning rate=0.0005) and
       categorical cross-entropy as the loss function.
    4. Normalizes the pixel values of the training and testing images to be
       between 0 and 1.
    5. Trains the network for 10 epochs with a batch size of 32.
    6. Evaluates the trained model on the test set, printing the final loss
       and accuracy.
    7. Visualizes the images that the model misclassified using an interactive
       matplotlib window. Users can browse through pages of misclassified
       images to inspect the model's errors.
"""

from src import nn
from src.data_utils import (
    load_mnist,
    normalize,
)
from src.visualizers import visualize_images

(x_train, y_train), (x_test, y_test) = load_mnist()

# I found the network to have better marginally better performance with 256
# units on the  first `Dense` layer instead of 128, but it came at the cost
# of 3x the training time (Processor: 12th Gen Intel(R) Core(TM) i7 - 12650H)
network = nn.NeuralNetwork("mnist")
network.add(nn.layers.Flatten(input_shape=(28, 28)))
network.add(nn.layers.Dense(128, 'relu'))
network.add(nn.layers.Dense(128, 'relu'))
network.add(nn.layers.Dense(10, 'softmax'))
network.compile(('adam', 0.0005), 'categorical_cross_entropy')

network.train(normalize(x_train), y_train, 32, 10)

loss, accuracy, misclassifications = network.evaluate(normalize(x_test), y_test)
print(f"\nEvaluation (x_test, y_test):\nLoss: {loss}, Accuracy: {accuracy}")
visualize_images(
    misclassifications['dataset'],
    misclassifications['labels'],
    misclassifications['predictions'],
    background_color='gray'
)
