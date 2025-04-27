# tests/test_data.py

import numpy as np
from rclass.data import load_mnist_train_and_test

def test_load_mnist_shapes():
    X_train, y_train, X_test, y_test = load_mnist_train_and_test()
    # MNIST should have 60 000 train and 10 000 test samples
    assert X_train.shape == (60000, 784)
    assert y_train.shape == (60000,)
    assert X_test.shape  == (10000, 784)
    assert y_test.shape  == (10000,)
    # Values should be floats
    assert X_train.dtype == np.float64
