from math import exp
import numpy as np

class LogisticRegressionClassifier:
    def __init__ (self, max_iter = 200, learning_rate = 0.01):
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def sigmoid (self, x):
        return 1 / (1 + exp(-x))
    
    def data_matrix (self, X):
        data_mat = []
        for d in X:
            data_mat.append([1.0, *d])
        return data_mat
    
    def fit (self, X, y):
        data_mat = self.data_matrix(X.values)
        self.weights = np.zeros((len(data_mat[0]), 1), dtype = np.float32)
        for iter_ in range(self.max_iter):
            for i in range(len(X)):
                result = self.sigmoid(np.dot(data_mat[i], self.weights))
                error = y[i] - result
                self.weights += self.learning_rate * error * np.transpose([data_mat[i]])
    
    def predict (self, X_test):
        data_mat = self.data_matrix(X_test.values)
        predictions = []
        for x in data_mat:
            result = np.dot(x, self.weights)
            if (result > 0):
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions