import numpy as np
from ml_utils import flatten

def softmax(X):
  expX = np.exp(X - np.max(X))
  return expX / expX.sum(axis=1, keepdims=True)

class LogisticRegression:
  def __init__(self, learning_rate, epochs=5, batch_size=256, verbose=True):
    self.learning_rate = learning_rate
    self.max_iter = epochs
    self.batch_size = batch_size
    self.verbose = verbose
    self.losses = []

  def fit(self, X, Y, teX, teY):
    X, Y, teX, teY = self.transform(X, Y, teX, teY)
    self.init_weight(X.shape[1], Y.shape[1])
    self.init_bias(Y.shape[1])
    self.losses = []
    pbar = tqdm(range(self.max_iter))
    itr = 0
    for epoch in pbar:
      for start in range(0, X.shape[0], self.batch_size):
        itr += 1
        end = min(X.shape[0], start + self.batch_size)
        X_b = X[start:end]
        Y_b = Y[start:end]
        out = self.forward(X_b)
        W_grad, b_grad = self.backward(out, X_b, Y_b)
        self.W -= self.learning_rate * W_grad
        self.b -= self.learning_rate * b_grad

        if itr % 100 == 0:
          loss, acc = self.evaluate(teX, teY)
          self.losses.append(loss)
          pbar.set_postfix({'epoch': epoch, 'iter':itr, 'loss': loss, 'Accuracy': acc})
          pbar.refresh()

    loss, acc = self.evaluate(teX, teY)
    pbar.set_postfix({'epoch': epoch, 'iter':itr, 'loss': loss, 'Accuracy': acc})
    pbar.refresh()
  
  def transform(self, X, Y, teX, teY):
    return flatten(X), to_onehot(Y), flatten(teX), to_onehot(teY)
  
  def evaluate(self, teX, teY):
    te_out = self.forward(teX)
    loss = self.loss(te_out, teY)
    labels_pred = np.argmax(te_out, axis=1)
    labels = np.argmax(teY, axis=1)
    acc = np.sum(labels_pred == labels) / len(labels)
    return loss, acc

  def loss(self, out, Y):
    return - np.sum(Y * np.log(out + 1e-8)) / Y.shape[0]

  def forward(self, X):
    return softmax(X @ self.W + self.b)

  def backward(self, out, X, Y):
    w_grad = X.T @ (out - Y) / Y.shape[0]
    b_grad = np.sum(out - Y, axis=0) / Y.shape[0]
    return w_grad, b_grad

  def predict(self, X):
    return np.argmax(self.forward(flatten(X)), axis=1) + 1

  def predict_proba(self, X):
    return self.forward(X)

  def init_weight(self, input_size, output_size):
    self.W = np.random.normal(0, 0.01, (input_size, output_size))

  def init_bias(self, output_size):
    self.b = np.random.normal(0, 0.01, (output_size, ))