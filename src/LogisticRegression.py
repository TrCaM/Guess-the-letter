"""
    Assignment1.LogisticRegression
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Logistic Regression implementation
"""
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def least_squares(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y


def cost_function(w, X, y):
    # Computes the cost function for all the training samples
    m = X.shape[0]
    probability = sigmoid(X @ w)
    total_cost = -(1 / m) * np.sum(y * np.log(probability) + (1 - y) * np.log(1 - probability))
    return total_cost


class LogisticRegression:
    """ The implementation of logistic regression
    """

    NAME = "Logistic Regression"

    def __init__(
        self,
        name="",
        learn_rate=0.01,
        tolerance=1e-2,
        max_iter=int(1e5),
        trans_func=None,
    ):
        self.weights = None
        self.name = name or self.NAME
        self.learn_rate = learn_rate
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.feature_transform_func = trans_func

    def gradient_descent(self, X, y, learn_rate, tolerance, max_iter):
        """ Gradient descent implementation

        @param X: the training matrix of examples
        @param y: the label vector
        @param learn_rate: the learning rate of GD
        @param tolerance: stopping tolerance condition (epsilon)
        @param max_iter: maximum iterations
        @return w: The trained weight vector
        """
        w = np.zeros(X.shape[1])
        step_norm = 0.000001
        self.cost = [cost_function(w, X, y)]
        for _ in range(max_iter):
            gradient = X.T @ (sigmoid(X @ w) - y) / y.size
            step = learn_rate * gradient
            w -= step
            cur_norm = np.linalg.norm(step, 2)
            step_norm = cur_norm
            self.cost.append(cost_function(w, X, y))
            if step_norm < tolerance:
                break
        return w

    def fit(self, train_data, labels):
        train_data = self.add_x_0(train_data)
        self.weights = self.gradient_descent(
            train_data, labels, self.learn_rate, self.tolerance, self.max_iter
        )
        # Uncomment to see the plot of cost function decrease by iterations
        # x = list(range(len(self.cost)))
        # y = self.cost
        # plt.plot(x, y)
        # plt.show()

    def add_x_0(self, X):
        """ Add the base feature x_0 with all having value of 1
        """
        X_new = np.ones((X.shape[0], X.shape[1] + 1))
        X_new[:, 1:] = X
        return X_new

    def predict(self, x_test):
        """ Predict the test data
        """
        x_test = self.add_x_0(x_test)
        y_predict = sigmoid(x_test @ self.weights) >= 0.5
        return y_predict.astype(int)

    @staticmethod
    def Acu_eval(predicted_labels, true_labels):
        """ Evaluate accuracy
        """
        return 1 - np.count_nonzero(predicted_labels ^ true_labels) / len(true_labels)

    def fit_and_predict(self, X_train, y_train, X_test, y_test):
        """ Aggregate function with the full train and test workflow
        """
        if self.feature_transform_func:
            X_train, X_test = self.feature_transform_func(X_train, X_test)

        self.fit(X_train, y_train)
        y_predict = self.predict(X_test)
        return self.Acu_eval(y_predict, y_test)
