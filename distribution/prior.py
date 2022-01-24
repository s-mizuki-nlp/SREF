#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os, sys, io
import warnings
from typing import Optional, Union, List, Any, Tuple, Dict
import pickle
import numpy as np
import scipy as sp

vector = np.array
matrix = np.ndarray
tensor = np.ndarray

from .continuous import MultiVariateNormal

class NormalInverseWishart(object):

    __EPS = 1E-5

    def __init__(self, vec_mu: vector, kappa: float, nu: float,
                 vec_phi: Optional[vector] = None, scalar_phi: Optional[float] = None):

        self._n_dim = len(vec_mu)
        self._mu = vec_mu
        if vec_phi is not None:
            self._diag_phi = vec_phi
            self._is_phi_diag = True
            self._is_phi_iso = False
        elif scalar_phi is not None:
            self._diag_phi = np.ones(shape=(self._n_dim,))*scalar_phi
            self._is_phi_diag = True
            self._is_phi_iso = True
        else:
            raise AttributeError("either `vec_phi` or `scalar_phi` must be specified.")

        self._kappa = kappa
        self._nu = nu

        self._validate()

    def _validate(self):
        msg = "dimensionality mismatch."
        assert len(self._mu) == self._n_dim, msg
        assert self._diag_phi.shape[0] == self._n_dim, msg

        assert self._nu > (self._n_dim + 1), f"`nu` must be greater than `n_dim` + 1: {self._nu} < {self._n_dim + 1}"

        return True

    @property
    def n_dim(self):
        return self._n_dim

    @property
    def is_phi_diag(self):
        return self._is_phi_diag

    @property
    def is_phi_iso(self):
        return self._is_phi_iso

    def posterior(self, mat_obs: matrix) -> "NormalInverseWishart":
        if mat_obs.ndim == 1:
            mat_obs = mat_obs.reshape((1, -1))
        n_obs, n_dim = mat_obs.shape
        assert n_dim == self._n_dim, f"dimensionality mismatch: {n_dim} != {self._n_dim}"

        # empirical mean
        vec_mu_e = mat_obs.mean(axis=0)

        # empirical covariance
        if n_obs == 1:
            diag_moment = np.zeros(shape=(n_dim,), dtype=np.float)
        else:
            # diag_moment = \sum_{i=1}^{n}(x_{ik}-\bar{x_{k}})^2
            diag_moment = np.sum((mat_obs - vec_mu_e)**2, axis=0)
        # if self.is_phi_diag:
        #     # take geometric mean
        #     scalar_cov = np.exp(np.mean(np.log(diag_moment)))
        #     diag_moment = np.ones(shape=(n_dim,)) * scalar_cov

        # \mu: weighted avg between \mu__0 and \mu_e
        vec_mu = (self._kappa * self._mu + n_obs * vec_mu_e) / (self._kappa + n_obs)

        # \kappa, \nu: simple update
        kappa = self._kappa + n_obs
        nu = self._nu + n_obs

        # \Phi but diagonal version
        # \delta \mu^{m} = {(\mu^e_{k} - \mu_0_k)^2}
        diag_mu_diff_moment  = (vec_mu_e - self._mu)**2
        diag_mu_diff_coef = (self._kappa * n_obs) / (self._kappa + n_obs)
        diag_phi = self._diag_phi + diag_moment + diag_mu_diff_coef * diag_mu_diff_moment

        return NormalInverseWishart(vec_mu=vec_mu, kappa=kappa, nu=nu, vec_phi=diag_phi)

    def approx_posterior_predictive(self, force_isotropic: bool = False) -> MultiVariateNormal:
        # exact posterior predictive is t-dist. We will approximate it with multivariate normal.
        vec_mu = self._mu
        diag_cov_coef = (self._kappa + 1) / (self._kappa * (self._nu - self.n_dim + 1))
        diag_cov = diag_cov_coef * self._diag_phi
        if force_isotropic:
            scalar_cov = np.exp(np.mean(np.log(diag_cov + 1E-15)))
            p_dist = MultiVariateNormal(vec_mu=vec_mu, scalar_cov=scalar_cov)
        else:
            p_dist = MultiVariateNormal(vec_mu=vec_mu, vec_cov=diag_cov)

        return p_dist