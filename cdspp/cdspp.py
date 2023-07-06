"""
    Python implementation of Cross Domain Structure Preserving Projection for Heterogeneous Domain Adaptation
    Author: Nils Paul Muttray, Youngjun Park

    Modified for Generalized Zero-Shot Domain Adaptaion with Pseudolabeling fit.
"""

import numpy as np
from cdspp.utils import *
from scipy import linalg
from scipy.spatial import KDTree
import math


# Model class for CDSPP
class CDSPP:
    def __init__(self, X_source, y_source, alpha, dim, missing=[]):
        """
        :param X_source: source domain data (features, samples)
        :param y_source: source domain label as numpy integers
        :param alpha: regularization parameter
        :param dim: latent space dimension
        :param missing: list of unknown classes
        """
        self.X_source = get_norm(X_source)
        self.y_source = y_source
        self.alpha = alpha
        self.dim = dim
        self.missing = missing


    def fit(self, X, y):
        """

        :param X: target domain data (features, samples)
        :param y: target domain data label as numpy integers
        :return: None
        """
        X = get_norm(X)
        W_s = get_W(self.y_source)
        W_t = get_W(y)
        W_c = get_W_cross(self.y_source, y)
        D_s = get_D_s(W_s)
        D_t = get_D_t(W_t)
        D_cs = get_D_s(W_c)
        D_ct = get_D_t(W_c)
        L_s = D_s - W_s + 0.5 * D_cs
        L_t = D_t - W_t + 0.5 * D_ct
        P = self.solve_P(X, W_c, L_s, L_t)
        self.P_source = P[:len(self.X_source)]
        self.P_target = P[len(self.X_source):]
        Z_s = self.P_source.T @ self.X_source
        Z_t = self.P_target.T @ X
        Z = np.concatenate((Z_s, Z_t), axis=1)
        Z, self.center = get_center(Z)
        Z = get_norm(Z)
        self.get_class_centers(Z, y)
        return


    # Solves the generalized eigenvalue problem
    def solve_P(self, X, W_c, L_s, L_t):
        """

        :param X: regularized target data
        :param W_c: cross-similarity matrix
        :param L_s: matrix as obtained by utils.py
        :param L_t: matrix as obtained by utils.py
        :return:
        """
        A_01 = self.X_source @ W_c @ X.T
        A_10 = X @ W_c.T @ self.X_source.T
        B_00 = self.X_source @ L_s @ self.X_source.T  # + np.identity(len(X))
        B_11 = X @ L_t @ X.T
        A = np.block([[np.zeros((A_01.shape[0], A_10.shape[1])), A_01],
                      [A_10, np.zeros((A_10.shape[0], A_01.shape[1]))]])
        B = np.block([[B_00, np.zeros((B_00.shape[0], B_11.shape[1]))],
                      [np.zeros((B_11.shape[0], B_00.shape[1])), B_11]])
        B = B + self.alpha * np.identity(len(B))
        eigvals, eigvecs = linalg.eigh(A, B)
        P = eigvecs[:, :-(self.dim + 1):-1]
        return P



    def transform_source(self):
        """

        :return: transformed stored source data
        """
        Z_s = self.P_source.T @ self.X_source
        Z_s = get_center(Z_s, self.center)
        Z_s = get_norm(Z_s)
        return Z_s


    def transform_target(self, X):
        """

        :param X: regularized target data
        :return: transformed target data
        """
        Z_t = self.P_target.T @ X
        Z_t = get_center(Z_t, self.center)
        Z_t = get_norm(Z_t)
        return Z_t


    def get_class_centers(self, Z, y):
        """

        :param Z: transformed concatenation of source and target data
        :param y: concatenated labels
        :return: None
        """
        Z = Z.T
        labels = np.concatenate((self.y_source, y))
        n_classes = np.amax(labels)+1
        means = np.zeros((n_classes, Z.shape[1]))
        for i in range(n_classes):
            tmp = Z[np.where(labels == i)]
            means[i] = np.mean(tmp, axis=0)
        means = means.T
        self.class_centers = get_norm(means)
        return


    def predict(self, X):
        """

        :param X: test target data
        :return: prediction
        """
        X = get_norm(X)
        Z = self.transform_target(X)
        Z = Z.T
        class_centres = self.class_centers.T
        tree = KDTree(class_centres)
        distances, y = tree.query(Z, k=1)
        return y


    def predict_with_distance(self, X):
        """

        :param X: test target data
        :return: prediction, distance to class center
        """
        X = get_norm(X)
        Z = self.transform_target(X)
        Z = Z.T
        class_centres = self.class_centers.T
        tree = KDTree(class_centres)
        distances, y = tree.query(Z, k=1)
        return y, distances


    # Fits the model semi-supervised
    def fit_semi_supervised(self, X_seen, X_unseen, y, K_seen=10, K_unseen=10, part=0.5):
        """

        :param X_seen: train data
        :param X_unseen: test data
        :param y: train labels
        :param K_seen: number of steps for pseudolabeling
        :param K_unseen: number of steps for pseudolabeling of unseen classes
        :param part: proportion of unseen class prediction during burn-in
        :return: None
        """
        self.fit(X_seen, y)

        for k in range(K_unseen):
            labels, distances = self.predict_with_distance(X_unseen)
            all_idx = np.arange(len(labels))
            if sum(np.in1d(labels, self.missing)) > 1:
                idx = np.array([])
                for i in self.missing:
                    temp = all_idx[np.in1d(labels, i)]
                    if len(temp) < 2:
                        continue
                    lowest_ind = np.argsort(distances[temp])[:math.ceil(len(temp) *((k + 1) / K_unseen) * part)]
                    idx = np.concatenate((idx, temp[lowest_ind]))
                seen = np.in1d(all_idx, idx)
                self.fit(np.concatenate((X_seen, X_unseen[:, seen]), axis=1), np.concatenate((y, labels[seen])))

        for k in range(K_seen):
            labels, distances = self.predict_with_distance(X_unseen)
            all_idx = np.arange(len(labels))
            lowest_ind = np.argsort(distances)[:math.ceil(len(distances) *((k + 1)/ K_seen))]
            idx = all_idx[lowest_ind]
            seen = np.in1d(all_idx, idx)
            self.fit(np.concatenate((X_seen, X_unseen[:, seen]), axis=1), np.concatenate((y, labels[seen])))

        return


    def get_params(self, deep=True):
        params = {"alpha": self.alpha, "dim": self.dim}
        return params
