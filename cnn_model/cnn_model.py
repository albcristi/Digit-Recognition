from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
import numpy as np
import matplotlib.pyplot as plt


class CnnModel:

    def __init__(self):
        '''
        We initialize the CNN model
        '''
        # create the model
        self.__model = Sequential()

        #  we add the required layers
        self.__model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
        self.__model.add(Conv2D(32, kernel_size=3, activation='relu'))
        self.__model.add(Flatten())  # turn img to 1D array
        self.__model.add(Dense(10, activation='softmax'))  # 10 because we have 10 classes

        #  compile the model
        self.__model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

        # result parameter after training the model
        self.__hist = None

    def train_model(self, x_train, y_train, x_test, y_test, no_epochs=3):
        '''
        This method will train our CNN based on the input
        training set and validate it based on the test set
        The evaluation results will be stored in the __hist
        field of our CnnModel class
        '''
        self.__hist = self.__model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=no_epochs)

    def get_history(self):
        '''
        This method will return the evaluation history
        after the model has been trained.
        In case our model is not yet trained, an run time
        error will be raised
        '''
        if self.__hist is None:
            raise RuntimeError("CnnMode is not trained yet")
        return self.__hist

    def predict(self, x_test):
        '''
        This method will return the predictions made by
        the model given some test values.
        We will return the index from the array having
        the best probability.
        For our digit prediction, the returned index from
        the array will represent the predicted digit
        '''
        return np.argmax(self.__model.predict(x_test), axis=1)


    def plot_results(self):
        if self.__hist is None:
            raise RuntimeError("CnnMode is not trained yet")
        # visualize the models accuracy
        plt.plot(self.__hist.history['acc'])
        plt.plot(self.__hist.history['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
