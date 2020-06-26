import numpy as np
from ml_utils import flatten

class SVDClassifier:
  def __init__(self, basis_images=20, verbose=True):
    self.verbose = verbose
    self.basis_images = basis_images
    self.num_classes = 0
    self.U_k = None

  def fit(self, X, Y, teX, teY):
    self.num_classes = len(np.unique(Y))
    X, Y, teX, teY = self.transform(X, Y, teX, teY)
    if self.verbose:
      print("Training using SVD decomposition...")
    self.U_k = np.array([
        np.linalg.svd(X[Y == label].T)[0][:, :self.basis_images].T
        for label in range(self.num_classes)
    ])
    if self.verbose:
      print("Finished!")
    acc = self.evaluate(teX, teY)
    if self.verbose:
      print(f'Accuracy: {acc*100:.4f}')
    return self

  def evaluate(self, teX, teY):
    preY = self.predict(teX)
    teY += 1
    return np.sum(test_y == preY) / len(test_y)

  def transform(self, X, Y, teX, teY):
    Y = Y - 1
    teY = teY - 1
    return flatten(X), Y, flatten(teX), teY

  def predict(self, X):
    X = flatten(X)
    residuals = np.zeros((X.shape[0], self.num_classes))
    for i in range(self.num_classes):
      U_i = self.U_k[i]
      residuals[:, i] = np.linalg.norm((np.identity(X.shape[1]) - U_i.T @ U_i) @ X.T, ord=2, axis=0)
    return np.argmin(residuals, axis=1) + 1