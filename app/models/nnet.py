import numpy as np
from ml_utils import flatten

def softmax(X):
  expX = np.exp(X - np.max(X))
  return expX / expX.sum(axis=1, keepdims=True)

def relu(X):
  return np.maximum(0, X)

def relu_back(Z):
  back = Z.copy()
  back[Z > 0] = 1
  return back

def sigmoid(Z):
  return 1/(1+np.exp(-Z))

def dSigmoid(s):
  return s * (1-s)

def softmax_backward(out):
  backT = (out.T @ np.ones((out.shape[0], out.shape[1]))) * (np.identity(out.shape[1]) - np.ones((out.shape[1], out.shape[0])) @ out)
  return backT.T

class NeuralNet:
  def __init__(self, hidden_layer_n, learning_rate, epochs=5, batch_size=256, verbose=True):
    self.learning_rate = learning_rate
    self.max_iter = epochs
    self.batch_size = batch_size
    self.verbose = verbose
    self.hidden_layer_n = 128
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
        z1, a1, z2, out = self.forward(X_b)
        W1_grad, b1_grad, W2_grad, b2_grad = self.backward(z1, a1, z2, out, X_b, Y_b)
        self.W1 -= self.learning_rate * W1_grad
        self.b1 -= self.learning_rate * b1_grad
        self.W2 -= self.learning_rate * W2_grad
        self.b2 -= self.learning_rate * b2_grad
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
    te_out = self.forward(teX)[3]
    loss = self.loss(te_out, teY)
    labels_pred = np.argmax(te_out, axis=1)
    labels = np.argmax(teY, axis=1)
    acc = np.sum(labels_pred == labels) / len(labels)
    return loss, acc

  def transform(self, X, Y, teX, teY):
    return flatten(X), to_onehot(Y), flatten(teX), to_onehot(teY)

  def loss(self, Y, out):
    return - np.sum(Y * np.log(out + 1e-8)) / Y.shape[0]

  def forward(self, X):
    z1 = X @ self.W1 + self.b1
    a1 = relu(z1)
    z2 = a1 @ self.W2 + self.b2
    return z1, a1, z2, softmax(z2)

  def backward(self, z1, a1, z2, out, X, Y):
    d_z2 = out - Y
    w2_grad = a1.T @ d_z2 / a1.shape[1]
    b2_grad = np.sum(d_z2, axis=0) / a1.shape[1]
    da1 = d_z2 @ self.W2.T
    d_z1  = da1 * relu_back(a1)
    w1_grad = X.T @ d_z1 / X.shape[1]
    b1_grad = np.sum(d_z1, axis=0) / X.shape[1]
    return w1_grad, b1_grad, w2_grad, b2_grad

  def predict(self, X):
    X = flatten(X)
    return np.argmax(self.forward(X)[3], axis=1) + 1

  def predict_proba(self, X):
    return self.forward(X)[3]

  def init_weight(self, input_size, output_size):
    self.W1 = np.random.normal(0, 0.01, (input_size, self.hidden_layer_n))
    self.W2 = np.random.normal(0, 0.01, (self.hidden_layer_n, output_size))

  def init_bias(self, output_size):
    self.b1 = np.random.normal(0, 0.01, (self.hidden_layer_n, ))
    self.b2 = np.random.normal(0, 0.01, (output_size, ))
