import logging
import numpy as np
import tensorflow as tf


def load_mnist_train_and_test():
    """
    Loads the MNIST dataset via tf.keras, flattens images to 784-vectors,
    and casts to float64 for numerical stability.
    Returns:
        X_train, y_train, X_test, y_test
    """
    logging.info("Loading MNIST dataset from tf.keras.datasets...")
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.reshape(-1, 28*28).astype(np.float64)
    X_test  = X_test.reshape(-1, 28*28).astype(np.float64)
    logging.info(f"Loaded shapes: X_train={X_train.shape}, y_train={y_train.shape};"
                 f" X_test={X_test.shape}, y_test={y_test.shape}")
    return X_train, y_train, X_test, y_test
