"""Softmax model."""

import numpy as np


class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray, batch) -> np.ndarray:
        """Calculate gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            gradient with respect to weights w; an array of same shape as w
        """
        w_b = np.zeros_like(self.w)
        for idx in batch:
            y_i = y_train[idx]
            if np.argmax(np.dot(self.w, X_train[idx])) == y_i:
                continue
            K = -np.max(np.dot(self.w, X_train[idx]))
            sum = np.sum(np.exp(np.dot(self.w, X_train[idx]) + K))
            for n in range(self.n_class):
                P_w = np.exp(np.dot(self.w[n], X_train[idx]) + K) / sum
                if y_i != n:
                    w_b[n] -= self.lr*P_w*X_train[idx]
                else:
                    P_y = np.exp(np.dot(self.w[y_i], X_train[idx]) + K) / sum
                    w_b[y_i] += self.lr*(1 - P_y)*X_train[idx]

        return w_b

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        self.w = np.zeros((self.n_class, X_train.shape[1]))
        self.w = np.column_stack((self.w, [1]*self.n_class))
        x_train = np.column_stack((X_train, [1]*X_train.shape[0]))

        b_size = 1000
        batch = np.arange(x_train.shape[0])

        for e in range(self.epochs):
            start = 0
            np.random.shuffle(batch)
            for b in range(int(x_train.shape[0] / b_size)):
                end = start + b_size
                self.w += self.calc_gradient(x_train, y_train, batch[start:end])
                start += b_size
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
        x_test = np.column_stack((X_test, [1]*X_test.shape[0]))
        for i, x in enumerate(x_test):
            y.append(int(np.argmax(np.dot(self.w, x))))

        return y
