import numpy as np
from ml_utils import flatten

class LDAClassifier:
  def __init__(self, verbose=True):
    self.verbose = verbose
    self.num_classes = 0
    self.mu = None
    self.sigma = None
    self.sigma_inv = None
    self.log_Py = None

  def fit(self, X, Y, teX, teY):
    X, Y, teX, teY = self.transform(X, Y, teX, teY)
    self.num_classes = len(np.unique(Y))
    if self.verbose:
      print("Calculating mu and covariance...")
    N = self.N_cal(Y)
    self.log_Py = self.log_Py_cal(N)
    self.mu = self.mu_cal(X, Y, N)
    self.sigma = self.covariance(X, Y, self.mu)
    self.sigma_inv = np.linalg.pinv(self.sigma)
    if self.verbose:
      print("Training Finished! Evaluating...")
    acc = self.evaluate(teX, teY)
    if self.verbose:
      print(f'Accuracy: {acc*100:.4f}')
    return self
  
  def evaluate(self, teX, teY):
    preY = self.predict(teX)
    teY += 1
    return np.sum(test_y == preY) / len(test_y)

  def transform(self, X, Y, teX, teY):
    X[X == 0] = 1e-8
    teX[teX == 0] = 1e-8
    Y = Y - 1
    teY = teY - 1
    return flatten(X), Y, flatten(teX), teY
  
  def log_Py_cal(self, N):
    return np.log(N / np.sum(N))

  def N_cal(self, Y):
    return np.bincount(Y)

  def mu_cal(self, X, Y, N):
    return np.array([np.sum(X[Y == i], axis=0) / N[i] for i in range(self.num_classes)])

  def covariance(self, X, Y, mu):
    sum_k = np.zeros((X.shape[1], X.shape[1]))
    for k in range(self.num_classes):
      X_k = X[Y == k]
      sum_k += (X_k - mu[k]).T @ (X_k - mu[k])
    return sum_k / (len(Y) - self.num_classes)

  def predict_row(self, x):
    theta = np.zeros(self.num_classes)
    for i, mu_k in enumerate(self.mu):
      theta[i] = x @ self.sigma_inv @ mu_k - 0.5 * mu_k @ self.sigma_inv @ mu_k + self.log_Py[i]
    return np.argmax(theta)

  def predict(self, X):
    return np.apply_along_axis(self.predict_row, 1, flatten(X)) + 1
