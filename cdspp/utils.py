import numpy as np


# l_2 normalisation of observations
def get_norm(X):
  l_2 = np.linalg.norm(X, axis=0)
  X_norm = np.divide(X, l_2)
  return X_norm


# either centers observations by calculating and returning mean or substracts given mean
def get_center(X, mean = None):
  try:
    if mean == None:
      mean = np.mean(X, axis = 1)
      X_centered = np.subtract(X.T, mean).T
      return X_centered, mean
  except:
    X_centered = np.subtract(X.T, mean).T
    return X_centered

# Creates different helper matrices

def get_W(labels):
  length = len(labels)
  W = np.zeros((length, length))
  for i in range(length):
    for j in range(i, length):
      if labels[i] == labels[j]:
        W[i,j] = 1
        W[j,i] = 1
  return W

def get_W_cross(labels_s, labels_t):
  W = np.zeros((len(labels_s), len(labels_t)))
  for i in range(len(labels_s)):
    for j in range(len(labels_t)):
      if labels_s[i] == labels_t[j]:
        W[i,j] = 1
  return W

def get_D_s(W):
  length = len(W)
  D = np.zeros((length, length))
  d = np.sum(W, axis=1)
  np.fill_diagonal(D, d)
  return D

def get_D_t(W):
  length = len(W[0])
  D = np.zeros((length, length))
  d = np.sum(W, axis=0)
  np.fill_diagonal(D, d)
  return D