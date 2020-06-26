import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import pickle
from tqdm.notebook import trange, tqdm

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
  train_df = pd.read_csv('emnist-letters-train.csv', header=None)
  test_df = pd.read_csv('emnist-letters-test.csv', header=None)

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

def flatten(X):
  return np.array([np.array(image).flatten('F') for image in X])

def get_prediction(test_img, model):
  test_img = test_img.reshape(1, test_img.shape[0], test_img.shape[1])
  return get_letter_from_label(model.predict(test_img)[0])

def display_prediction(model, test_img, true_label=None):
  pred_letter = get_prediction(test_img, model)
  if true_label:
    true_letter = get_letter_from_label(true_label)
  plt.imshow(test_img, cmap='gray')
  mess = f'Prediction: {pred_letter}'
  plt.title(mess)
  if true_label:
    mess += f' - Correct answer: {true_letter}'
    plt.title(mess, fontdict={'color': 'green' if true_letter == pred_letter else 'red' })
  
def to_onehot(Y):
  return np.eye(26)[Y - 1]

def save_model(model, name):
  filename = f'{name}.sav'
  pickle.dump(model, open(filename, 'wb'))

def load_model(filename):
  return pickle.load(open(filename, 'rb'))