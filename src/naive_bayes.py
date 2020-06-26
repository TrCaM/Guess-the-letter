import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.sparse import csr_matrix, save_npz, load_npz, vstack

# TODO: axis = 1 => row wise
# TODO: axis = 0 => col wise

class MyNaiveBayes(BaseEstimator, ClassifierMixin):

    def fit(self, X, y):
        theta_y = sum(y) / len(y)
        sum_X = np.array([np.array(np.sum(X[y == k], axis=0)).flatten() for k in range(2)]).T
        sum_y = np.array([len(y) - sum(y), sum(y)])
        self.theta_X = (sum_X + 1) / (sum_y + 2)
        self.log_1 = np.log(theta_y / (1 - theta_y))
        self.log_2 = np.log(self.theta_X[:, 1] / self.theta_X[:, 0])
        self.log_3 = np.log((1 - self.theta_X[:, 1]) / (1 - self.theta_X[:, 0]))
        return self

    def predict(self, X):
        X_arr = X.toarray()
        self.logits = np.full(X.shape[0], self.log_1) + X @ self.log_2 + (1 - X_arr) @ self.log_3
        return (self.logits > 0).astype(int)


    def predict_proba(self, X):
        pred = np.random.rand(X.shape[0], self.classes_.size)
        return pred / np.sum(pred, axis=1)[:, np.newaxis]


def store_sparse_matrix():
    print("Reading data...")
    data1 = pd.read_csv('./datasets/count_data_binary/test_transformed.csv')
    data2 = pd.read_csv('./datasets/count_data_binary/train_transformed.csv')
    X1, y1 = csr_matrix(data1.drop('y_label', axis=1).values), data1['y_label'].values
    X2, y2 = csr_matrix(data2.drop('y_label', axis=1).values), data2['y_label'].values
    X = vstack((X1, X2))
    y = np.concatenate((y1, y2))
    save_npz("./datasets/movies_X.npz", X)
    np.save('./datasets/movies_Y.npy', y)


def load_X_y():
    X = load_npz("./datasets/movies_X.npz")
    y = np.load("./datasets/movies_Y.npy")
    return X, y


def try_LR(X, y):
    pass


if __name__ == "__main__":
    X, y = load_X_y()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    clf = MyNaiveBayes()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

