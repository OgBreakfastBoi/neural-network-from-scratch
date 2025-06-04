from data_utils import (
    load_mnist,
    normalize,
)
import nn

network = nn.NeuralNetwork((28, 28))
network.add(nn.Dense(128, 'relu'))
network.add(nn.Dense(128, 'relu'))
network.add(nn.Dense(10, 'softmax'))
network.compile('adam', 'cross_categorical_entropy_loss')

(x_train, y_train), (x_test, y_test) = load_mnist()

m = network.run(normalize(x_test), y_test, True) # feed forward test with random weights on MNIST
