"""Logistic regression model."""

import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.threshold = 0.5

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """

        return 1 / (1 + np.exp(-z))

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        self.w = np.zeros(X_train.shape[1])

        for e in range(self.epochs):
            for i, x in enumerate(X_train):
                y_i = y_train[i]
                y = self.sigmoid(np.dot(self.w, x)) > self.threshold
                if y != y_i:
                    if y_i == 0:
                        y_i = -1
                    self.w += self.lr*self.sigmoid(-y_i*np.dot(self.w, x))*y_i*x
            self.lr *= np.exp(-(e+1))

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
        for i, x in enumerate(X_test):
            y.append(np.dot(self.w, x) > self.threshold)

        return y
