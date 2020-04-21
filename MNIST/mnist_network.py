"""Bridging code between the MNIST data format and the vector
representation used in the network class.

"""

import mnist
import numpy as np


def vectorize_data(data):
    """Maps MNIST data to a pair of matrices. An MNIST data point is a
    pair consisting of a 784-component tuple of integers between 0 and
    255 (representing greyscale values) and an integer between 0 and 9
    (representing the target digit). The resulting matrices contain a
    row vector for each data point. Greyscale values are normalized to
    the interval [0, 1]. Labels are represented as one-hot vectors.

    """
    xs, ys = [], []
    for image, label in data:
        xs.append([pixel / 255 for pixel in image])
        ys.append([float(digit == label) for digit in range(10)])
    return np.array(xs), np.array(ys)


def read_training_data():
    """Reads training data. This returns a pair "(x, y)" of matrices where
    "x" is a "numpy.ndarray" of shape (60000, 784) whose rows
    represent images and "y" is a "numpy.ndarray" of shape (60000, 10)
    whose rows are one-hot vectors representing digits.

    """
    return vectorize_data(mnist.read_training_data())


def read_test_data():
    """Reads test data. This returns a pair "(x, y)" of matrices where "x"
    is a "numpy.ndarray" of shape (10000, 784) whose rows represent
    images and "y" is a "numpy.ndarray" of shape (10000, 10) whose
    rows are one-hot row vector representing digits.

    """
    return vectorize_data(mnist.read_test_data())


if __name__ == "__main__":
    x, y = read_training_data()
    print(x.shape, y.shape)
