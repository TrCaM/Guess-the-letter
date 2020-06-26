import numpy as np
from ml_utils import flatten
from tqdm.notebook import trange, tqdm

class KNNClassifier:
  def __init__(self, k=20, verbose=True, eval_total=1000):
    self.verbose = verbose
    self.k = k
    self.num_classes = 0
    self.X = None
    self.Y = None
    self.evals = 1000

  def fit(self, X, Y, teX, teY, eval=True):
    self.num_classes = len(np.unique(Y))
    self.X, self.Y, teX, teY = self.transform(X, Y, teX, teY)
    if self.verbose:
      print("Stored all data for predictions!")
    if eval:
      acc = self.evaluate(teX, teY, self.evals)
      if self.verbose:
        print(f'Accuracy: {acc*100:.4f}')

  def evaluate(self, teX, teY, pred_total):
    idx = np.arange(teY.shape[0])
    np.random.shuffle(idx)
    teX = teX[idx]
    teY = teY[idx]
    teY += 1
    preY = self.predict(teX[:pred_total])
    return np.sum(teY[:pred_total] == preY) / pred_total

  def transform(self, X, Y, teX, teY):
    Y = Y - 1
    teY = teY - 1
    return flatten(X), Y, flatten(teX), teY

  def distances(self, z):
    return np.linalg.norm(self.X - z.reshape((1, 784)), ord=2, axis=1)

  def find_knn(self, z):
    distances = self.distances(z)
    return np.argpartition(distances, self.k)[:self.k]

  def predict(self, Z):
    Z = flatten(Z)
    y_pred = np.zeros(Z.shape[0])
    if self.verbose:
      print("Predicting...")
    corrects = 0
    for i, z in tqdm(enumerate(Z), total=Z.shape[0]):
      knn_idx = self.find_knn(z)
      knn_labels = self.Y[knn_idx]
      y_pred[i] = np.bincount(knn_labels).argmax() + 1
    return y_pred.astype(int)
