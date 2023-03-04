"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in Lecture 3.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        self.w = np.random.rand(self.n_class, X_train.shape[1])
        self.w = np.column_stack((self.w, [1]*self.n_class))
        x_train = np.column_stack((X_train, [1]*X_train.shape[0]))

        for e in range(self.epochs):
            for i, x in enumerate(x_train):
                y_i = y_train[i]
                if np.argmax(np.dot(self.w, x)) == y_i:
                    continue
                w_y = np.dot(self.w[y_i], x)
                for n in range(self.n_class):
                    if y_i != n and np.dot(self.w[n], x) > w_y:
                        self.w[y_i] += self.lr*x
                        self.w[n] -= self.lr*x
            self.lr *= np.exp(-e)

        return

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        y = []
        x_test = np.column_stack((X_test, [1]*X_test.shape[0]))
        for i, x in enumerate(x_test):
            y.append(int(np.argmax(np.dot(self.w, x))))

        return y
