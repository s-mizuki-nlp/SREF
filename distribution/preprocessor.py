#!/usr/bin/env python
# -*- coding:utf-8 -*-

import io, pickle, json
from typing import Optional, Tuple, Dict, Any
import numpy as np

vector = np.array
matrix = np.ndarray
tensor = np.ndarray

from .utils import l2_normalize


class WhiteningPreprocessor(object):

    def __init__(self, pre_norm: bool = False, post_norm: bool = False, n_dim_reduced: int = None,
                 vec_mu: Optional[vector] = None, mat_u: Optional[matrix] = None, vec_lambda: Optional[vector] = None):
        """
        Whitening algorithm proposed in WhiteingBERT [Huang+, EMNLP2021]. It also supports dimensionality reduction.

        :param pre_norm: L2 normalization before transformation.
        :param post_norm: L2 normalizaiton after transformation
        :param n_dim_reduced: (optional) reduced number of dimensions. specify None when disabled.
        :param vec_mu: (optional) mean vector of observations.
        :param mat_u: (optional) eigenvectors of the covariance matrix.
        :param vec_lambda: (optional) eigenvalues of the covariance matrix.
        """
        self._pre_norm = pre_norm
        self._post_norm = post_norm
        self._n_dim_reduced = n_dim_reduced
        self._vec_mu = vec_mu
        self._mat_u = mat_u
        self._vec_lambda = vec_lambda

    def serialize(self):
        param_names = ("pre_norm","post_norm","n_dim_reduced","vec_mu","mat_u","vec_lambda")
        dict_ret = {}
        for param_name in param_names:
            dict_ret[param_name] = self.__getattribute__("_" + param_name)
        return dict_ret

    @classmethod
    def deserialize(cls, object: Dict[str, Any]):
        return cls(**object)

    @classmethod
    def load(cls, file_path: str) -> "WhiteningPreprocessor":
        with io.open(file_path, mode="rb") as ifs:
            obj = pickle.load(ifs)
            model = cls.deserialize(obj)
        return model

    def fit(self, mat_obs: matrix):
        if mat_obs.ndim == 1:
            mat_obs = mat_obs.reshape((1, -1))

        # pre L2 normalization
        if self._pre_norm:
            mat_obs = l2_normalize(mat_obs)

        n_obs, n_dim = mat_obs.shape
        assert n_obs > 1, f"sample size must be greater than 1."

        # empirical mean
        vec_mu = np.mean(mat_obs, axis=0)

        # empirical covariance
        # mat_cov: (n_dim, n_dim)
        mat_cov = (mat_obs - vec_mu).T.dot(mat_obs - vec_mu)

        # top-k eigenvectors and eigenvalues
        n_dim_reduced = n_dim if self._n_dim_reduced is None else self._n_dim_reduced
        # mat_u: (n_dim, n_dim_reduced), vec_lambda: (n_dim_reduced,)
        mat_u, vec_lambda = self._calc_principal_component_vectors_and_eigenvalues(mat_cov=mat_cov, n_dim_reduced=n_dim_reduced)

        # set parameters
        self._n_dim_reduced = n_dim_reduced
        self._vec_mu = vec_mu
        self._mat_u = mat_u
        self._vec_lambda = vec_lambda

    def transform(self, mat_obs: matrix) -> np.ndarray:
        if not hasattr(self, "_factor_loading_matrix"):
            d_sqrt_inv = 1. / np.sqrt(self._vec_lambda).reshape(1, -1)
            self._factor_loading_matrix = self._mat_u * d_sqrt_inv

        if self._pre_norm:
            mat_obs = l2_normalize(mat_obs)

        # mat_trans: (n_obs, n_dim_reduced)
        mat_trans = (mat_obs - self._vec_mu).dot(self._factor_loading_matrix)

        if self._post_norm:
            mat_trans = l2_normalize(mat_trans)

        return mat_trans

    def _calc_principal_component_vectors_and_eigenvalues(self, mat_cov: matrix, n_dim_reduced: int) -> Tuple[matrix, vector]:
        n_dim = mat_cov.shape[0]
        assert n_dim_reduced <= n_dim, "reduced dimension size `n_dim_reduced` must be smaller than original dimension size."

        # eigen decomposition
        vec_l, mat_w = np.linalg.eig(mat_cov)
        # take largest top-k eigenvectors
        idx_rank = vec_l.argsort()[::-1]
        vec_l_h = vec_l[idx_rank[:n_dim_reduced]]
        mat_w_h = mat_w[:, idx_rank[:n_dim_reduced]]

        # returned mat_w_h is the eigenvectors. shape will be (n_dim, n_dim_reduced). each column is a i-th greatest eigenvector.
        # returned vec_l_h is the eigenvalues. shape will be (n_dim_reduced). i-th element is the i-th greatest eigenvalue.
        return mat_w_h, vec_l_h

    def __str__(self):
        param_names = ("pre_norm","post_norm","n_dim_reduced")
        dict_ret = {}
        for param_name in param_names:
            dict_ret[param_name] = self.__getattribute__("_" + param_name)
        return json.dumps(dict_ret)

    @property
    def n_dim_reduced(self):
        return self._n_dim_reduced