from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


def load_data_set():
    '''
    This function will return the training
    and testing data set for the cnn
    '''
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return x_train, y_train, x_test, y_test


def reshape_data_set(to_be_reshaped, n):
    '''
    Reshape the dataset
    '''
    # n  - no of entries
    # each img will be 28X28 pixels
    # 1 for gray scale
    to_be_reshaped = to_be_reshaped.reshape(n, 28, 28, 1)
    return to_be_reshaped


def hot_encoding(to_be_encoded):
    result = to_categorical(to_be_encoded)
    return result



