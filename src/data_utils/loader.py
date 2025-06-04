from numpy import ndarray
from tensorflow.keras.datasets.mnist import load_data


def load_mnist() -> tuple[tuple[ndarray, ndarray], tuple[ndarray, ndarray]]:
    """
    Loads the MNIST dataset.

    This is a dataset of 60,000 28x28 grayscale images of the 10 digits,
    along with a test set of 10,000 images.
    More info can be found at the
    [MNIST homepage](http://yann.lecun.com/exdb/mnist/).

    Returns:
        A Tuple of NumPy arrays `(x_train, y_train), (x_test, y_test)`.

        ``x_train``: `uint8` NumPy array of grayscale image data with shapes
          `(60000, 28, 28)`, containing the training data. Pixel values range
          from 0 to 255.

        ``y_train``: `uint8` NumPy array of digit labels (integers in range 0-9)
          with shape `(60000,)` for the training data.

        ``x_test``: `uint8` NumPy array of grayscale image data with shapes
          `(10000, 28, 28)`, containing the test data. Pixel values range
          from 0 to 255.

        ``y_test``: `uint8` NumPy array of digit labels (integers in range 0-9)
          with shape `(10000,)` for the test data.
    """

    return load_data()
