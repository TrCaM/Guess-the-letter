# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Importing libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# # Utility functions

# %%
def fix_image(image):
  """
  Edit the original image data so to make them look natural
  """
  image = image.reshape([28, 28])
  image = np.fliplr(image)
  image = np.rot90(image)
  return image

def load_data(fix=False, one_hot_label=False, binary=False):
  """
  Loading all the train and test datasets
  """
  train_df = pd.read_csv('./data/emnist-letters-train.csv', header=None)
  test_df = pd.read_csv('./data/emnist-letters-test.csv', header=None)

  train_X = train_df.iloc[:, 1:].to_numpy()
  test_X = test_df.iloc[:, 1:].to_numpy()
  if binary:
    train_X[train_X > 0] = 1
    test_X[test_X > 0] = 1
  if fix:
    train_X = np.apply_along_axis(fix_image, 1, train_X)
    test_X = np.apply_along_axis(fix_image, 1, test_X)

  train_y = train_df.iloc[:, 0].to_numpy()
  test_y = test_df.iloc[:, 0].to_numpy()
  if one_hot_label:
    test_y = np.eye(26)[test_y - 1]
    train_y = np.eye(26)[train_y - 1]
  return (train_X, train_y), (test_X, test_y)

def get_letter_from_label(label):
  return chr(ord('a') + label - 1)

# %% [markdown]
# # Investigating datasets

# %%
(train_X, train_y), (test_X, test_y) = load_data(True)

# %% [markdown]
# ## Statistic on classes

# %%
labels, count = np.unique(train_y, return_counts=True)
labels = list(map(get_letter_from_label, labels))
class_count = dict(zip(labels, count))
print(class_count)
print(np.unique(train_y-1, return_counts=True))


# %% [markdown]
# ## Display examples for a list of examples

# %%
# %matplotlib
fig, axs = plt.subplots(5, 6, figsize=(10, 10))
for i, (x, y) in enumerate(zip(train_X[:30], train_y[:30])):
  ax = axs[i // 6, i % 6]
  ax.imshow(x)
  ax.set_title(get_letter_from_label(y))
  ax.set_xticks([])
  ax.set_yticks([])
plt.tight_layout()
plt.show()

# %% [markdown]
# # Multinomial Logistic Regression

# %% [markdown]
# ## Introduction
# TODO: Working on proving the gradient step by step

# %% [markdown]
# ## Proof of concept

# %% [markdown]
# ## Implementation

# %%
def softmax(X):
  expX = np.exp(X - np.max(X))
  return expX / expX.sum(axis=0, keepdims=True)

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

class LogisticRegression:
  def __init__(self, hidden_layer_n, learning_rate, epochs=5, batch_size=256, verbose=True):
    self.learning_rate = learning_rate
    self.max_iter = epochs
    self.batch_size = batch_size
    self.verbose = verbose
    self.hidden_layer_n = 128
    self.losses = []

  def fit(self, X, Y, teX, teY):
    self.init_weight(X.shape[1], Y.shape[1])
    self.init_bias(Y.shape[1])
    itr = 0
    self.losses = []
    for epoch in range(self.max_iter):
      for start in range(0, X.shape[0], self.batch_size):
        itr += 1
        end = min(X.shape[0], start + self.batch_size)
        X_b = X[start:end]
        Y_b = Y[start:end]
        z1, a1, a2, out = self.forward(X_b)
        W1_grad, b1_grad, W2_grad, b2_grad = self.backward(z1, a1, a2, out, X_b, Y_b)
        self.W1 -= self.learning_rate * W1_grad
        self.b1 -= self.learning_rate * b1_grad
        self.W2 -= self.learning_rate * W2_grad
        self.b2 -= self.learning_rate * b2_grad
        if itr % 100 == 1:
          te_out = self.forward(teX)
          loss = self.loss(te_out, teY)
          labels_pred = np.argmax(te_out, axis=1)
          labels = np.argmax(teY, axis=1)
          acc = np.sum(labels_pred == labels) / len(labels)
          self.losses.append(loss)
          if self.verbose:
            print(f'Epoch {epoch} - Iter {itr}\'s loss: {loss:.4f} - Accuracy: {acc:.4f}')
    return self

  def loss(self, out, Y):
    return - np.sum(Y * np.log(out + 1e-8)) / Y.shape[0]

  def forward(self, X):
    z1 = X @ self.W1 + self.b1
    a1 = relu(z1)
    a2 =  sigmoid(a1 @ self.W2 + self.b2)
    return z1, a1, a2, softmax(a2)

  def backward(self, z1, a1, a2, out, X, Y):
    d_loss = - (Y / out - (1 - Y) - (1 - Y) / (1- out))
    d_z2 = d_loss * dSigmoid(out)
    w2_grad = a1.T @ d_z2  / a1.shape[1]
    b2_grad = np.sum(d_z2, axis=0) / a1.shape[1]
    da1 = self.W2.T @ d_z2
    dz1  = da1 * relu_back(a1)
    din = self.W1.T @ dz1
    w1_grad = X.T @ din / X.shape[1]
    b1_grad = np.sum(din, axis=0) / X.shape[1]
    return w1_grad, b1_grad, w2_grad, b2_grad

  def predict(self, X):
    return np.argmax(self.forward(X), axis=1)

  def predict_proba(self, X):
    return self.forward(X)

  def init_weight(self, input_size, output_size):
    self.W1 = np.random.normal(0, 0.01, (input_size, self.hidden_layer_n))
    self.W2 = np.random.normal(0, 0.01, (self.hidden_layer_n, output_size))

  def init_bias(self, output_size):
    self.b1 = np.random.normal(0, 0.01, (self.hidden_layer_n, ))
    self.b2 = np.random.normal(0, 0.01, (output_size, ))

# %% [markdown]
# ## Training and Result

# %%
# Reload data which already at their flatten form
(train_X, train_y), (test_X, test_y) = load_data(fix=False, one_hot_label=True)
# train_X = train_X / 255
# test_X = test_X / 255
print(train_X.shape)
print(train_y.shape)
print(test_X.shape)
print(test_y.shape)

# %%
LR = LogisticRegression(0.001, epochs=3)
LR.fit(train_X, train_y, test_X, test_y)


# %% [markdown]
# # Multinomial LDA

# %%
class LDAClassifier:
  def __init__(self, verbose=True):
    self.verbose = verbose
    self.num_classes = 0
    self.mu = None
    self.sigma = None
    self.sigma_inv = None
    self.log_Py = None

  def fit(self, X, Y):
    self.num_classes = len(np.unique(Y))
    X = self.transform(X)
    if self.verbose:
      print("Calculating mu and covariance...")
    N = self.N_cal(Y)
    self.log_Py = self.log_Py_cal(N)
    self.mu = self.mu_cal(X, Y, N)
    self.sigma = self.covariance(X, Y, self.mu)
    self.sigma_inv = np.linalg.pinv(self.sigma)
    if self.verbose:
      print("Finished!")
    return self

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

  def transform(self, X):
    return np.array([np.array(image).flatten('F') for image in X])

  def predict_row(self, x):
    theta = np.zeros(self.num_classes)
    for i, mu_k in enumerate(self.mu):
      theta[i] = x @ self.sigma_inv @ mu_k - 0.5 * mu_k @ self.sigma_inv @ mu_k + self.log_Py
    return np.argmax(theta)

  def predict(self, X):
    X = self.transform(X)
    return np.apply_along_axis(self.predict_row, 1, X)

# %%
(train_X, train_y), (test_X, test_y) = load_data(fix=True, one_hot_label=False)
train_y -= 1
test_y -= 1
train_X = train_X / 255.0
test_X = test_X / 255.0

# %%
lda = LDAClassifier()
lda.fit(train_X, train_y)
y_pred = lda.predict(test_X)
np.sum(test_y == y_pred) / len(test_y)

# %% [markdown]
# # SVD method

# %%
(train_X, train_y), (test_X, test_y) = load_data(fix=True, one_hot_label=False)
train_y -= 1
test_y -= 1
train_X = train_X / 255.0
test_X = test_X / 255.0

# flat =  np.array([np.array(image).flatten('F') for image in train_X])
# U, d, Vt = np.linalg.svd(flat[train_y == 0].T)[0]
# Uk = U[:, :20]
# Uk.shape
# print("done")


# %%
class SVDClassifier:
  def __init__(self, basis_images=20, verbose=True):
    self.verbose = verbose
    self.basis_images = basis_images
    self.num_classes = 0
    self.U_k = None

  def fit(self, X, Y):
    self.num_classes = len(np.unique(Y))
    X = self.transform(X)
    if self.verbose:
      print("Training using SVD decomposition...")
    self.U_k = np.array([
        np.linalg.svd(X[Y == label].T)[0][:, :self.basis_images].T
        for label in range(self.num_classes)
    ])
    if self.verbose:
      print("Finished!")
    return self

  def transform(self, X):
    return np.array([np.array(image).flatten('F') for image in X])

  def predict(self, X):
    X = self.transform(X)
    residuals = np.zeros((X.shape[0], self.num_classes))
    for i in range(self.num_classes):
      U_i = self.U_k[i]
      residuals[:, i] = np.linalg.norm((np.identity(X.shape[1]) - U_i.T @ U_i) @ X.T, ord=2, axis=0)
    return np.argmin(residuals, axis=1)

  def predict_proba(self, X):
    return 0

# %%

svd = SVDClassifier()
svd.fit(train_X, train_y)
svd.U_k.shape
y_pred = svd.predict(test_X)
np.sum(test_y == y_pred) / len(test_y)

# %% [markdown]
# # K Nearest Neighbors

# %%
class KNNClassifier:
  def __init__(self, k=20, verbose=True):
    self.verbose = verbose
    self.k = k
    self.num_classes = 0
    self.X = None
    self.Y = None

  def fit(self, X, Y):
    self.num_classes = len(np.unique(Y))
    self.X = self.transform(X)
    self.Y = Y
    if self.verbose:
      print("Stored all data for predictions!")
    return self

  def transform(self, X):
    return np.array([np.array(image).flatten('F') for image in X])

  def distances(self, z):
    return np.linalg.norm(self.X - z.reshape((1, 784)), ord=2, axis=1)

  def find_knn(self, z):
    distances = self.distances(z)
    return np.argpartition(distances, self.k)[:self.k]

  def predict(self, Z):
    Z = self.transform(Z)
    y_pred = np.zeros(Z.shape[0])
    if self.verbose:
      print("Predicting...")
    for i, z in enumerate(Z):
      knn_idx = self.find_knn(z)
      knn_labels = self.Y[knn_idx]
      y_pred[i] = np.bincount(knn_labels).argmax()
      if self.verbose and i % 200 == 0:
        print(f'done {i+1} predictions...')
    return y_pred

# %%
(train_X, train_y), (test_X, test_y) = load_data(fix=True, one_hot_label=False)
train_y -= 1
test_y -= 1
train_X = train_X / 255.0
test_X = test_X / 255.0

# %%
knn = KNNClassifier()
knn.fit(train_X, train_y)
y_pred = knn.predict(test_X)
np.sum(test_y == y_pred) / len(test_y)

