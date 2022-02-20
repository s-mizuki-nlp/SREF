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
    _AVAILABLE_POSTERIOR_INFERENCE_METHOD = ["default", "known_dof", "known_variance", "predictive_posterior", "mean_posterior"]

    def __init__(self, vec_mu: vector, kappa: float, nu: float,
                 posterior_inference_method: str,
                 vec_phi: Optional[vector] = None, scalar_phi: Optional[float] = None):

        assert posterior_inference_method in self._AVAILABLE_POSTERIOR_INFERENCE_METHOD, \
            f"invalid `posterior_inference_method` value. valid values are: {self._AVAILABLE_POSTERIOR_INFERENCE_METHOD}"
        if posterior_inference_method in ("known_dof",):
            warnings.warn(f"`{posterior_inference_method}` isn't recommended because covariance will diverge when \kappa or \nu - DoF is small compared to 1.0.")
            warnings.warn(f"We recommend using `mean_posterior` or `predictive_posterior` instead.")

        self._n_dim = len(vec_mu)
        self._posterior_inference_method = posterior_inference_method
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

    @classmethod
    def AVAILABLE_POSTERIOR_INFERENCE_METHOD(cls):
        return cls._AVAILABLE_POSTERIOR_INFERENCE_METHOD

    @property
    def n_dim(self):
        return self._n_dim

    @property
    def is_phi_diag(self):
        return self._is_phi_diag

    @property
    def is_phi_iso(self):
        return self._is_phi_iso

    def _posterior_known_variance(self, mat_obs: matrix, sample_weights: Optional[vector] = None) -> "NormalInverseWishart":
        # Theoretically conjugate prior of the multivariate normal with known covariance is the multivariate normal (not normal inverse-wishart).
        # In order to simplify the implementation, we alternatively use NIW with \kappa_0 = \nu_0 = 1.0 as the conjugate prior.
        n_obs, n_dim = mat_obs.shape

        # dummy DoF parameters
        kappa_dash = 1.0
        nu_dash = 1.0 + n_dim + 1

        # empirical mean
        if sample_weights is None:
            vec_mu_e = mat_obs.mean(axis=0)
        else:
            # E[X] = \sum_{i}p(x_i)x_i
            # note: sample_weights are normalized
            vec_mu_e = np.sum(mat_obs * sample_weights.reshape(n_obs,1), axis=0)

        # empirical variance as the known variance
        if n_obs == 1:
            vec_diag_cov_e = np.zeros(shape=(n_dim,), dtype=np.float)
        else:
            # diag_moment = \sum_{i=1}^{n}(x_{ik}-\bar{x_{k}})^2
            if sample_weights is None:
                vec_diag_cov_e = np.sum((mat_obs - vec_mu_e)**2, axis=0) / (n_obs - 1)
            else:
                vec_diag_cov_e = np.sum( ((mat_obs - vec_mu_e)**2) * n_obs * sample_weights.reshape(n_obs,1), axis=0) / (n_obs - 1)

        # prior mean and variance
        vec_mu_0, vec_diag_cov_0 = self.mean

        # posterior mean and variance using diagonal approximation
        # \Sigma'
        vec_diag_cov_dash = vec_diag_cov_0 * vec_diag_cov_e / (vec_diag_cov_e + n_obs * vec_diag_cov_0)
        # \mu'
        vec_mu_dash = (vec_diag_cov_e * vec_mu_0 + n_obs * vec_diag_cov_0 * vec_mu_e) / (vec_diag_cov_e + n_obs * vec_diag_cov_0)
        vec_diag_phi = vec_diag_cov_dash * (nu_dash - n_dim - 1) # = \Sigma'

        return NormalInverseWishart(vec_mu=vec_mu_dash, kappa=kappa_dash, nu=nu_dash, vec_phi=vec_diag_phi,
                                    posterior_inference_method=self._posterior_inference_method)

    def _posterior_default(self, mat_obs: matrix, sample_weights: Optional[vector] = None,
                           kappa_dash: Optional[float] = None, nu_dash: Optional[float] = None) -> "NormalInverseWishart":

        n_obs, n_dim = mat_obs.shape

        # empirical mean
        if sample_weights is None:
            vec_mu_e = mat_obs.mean(axis=0)
        else:
            # E[X] = \sum_{i}p(x_i)x_i
            vec_mu_e = np.sum(mat_obs * sample_weights.reshape(n_obs,1), axis=0)

        # empirical covariance
        if n_obs == 1:
            diag_moment = np.zeros(shape=(n_dim,), dtype=np.float)
        else:
            # diag_moment = \sum_{i=1}^{n}(x_{ik}-\bar{x_{k}})^2
            if sample_weights is None:
                diag_moment = np.sum((mat_obs - vec_mu_e)**2, axis=0)
            else:
                diag_moment = np.sum( ((mat_obs - vec_mu_e)**2) * n_obs * sample_weights.reshape(n_obs,1), axis=0)

        # \mu: weighted avg between \mu__0 and \mu_e
        vec_mu = (self._kappa * self._mu + n_obs * vec_mu_e) / (self._kappa + n_obs)

        # \kappa, \nu: simple update
        if kappa_dash is None:
            _kappa = self._kappa + n_obs
        else:
            _kappa = kappa_dash
        if nu_dash is None:
            _nu = self._nu + n_obs
        else:
            _nu = nu_dash

        # \Phi but diagonal version
        # \delta \mu^{m} = {(\mu^e_{k} - \mu_0_k)^2}
        diag_mu_diff_moment  = (vec_mu_e - self._mu)**2
        diag_mu_diff_coef = (self._kappa * n_obs) / (self._kappa + n_obs)
        diag_phi = self._diag_phi + diag_moment + diag_mu_diff_coef * diag_mu_diff_moment

        return NormalInverseWishart(vec_mu=vec_mu, kappa=_kappa, nu=_nu, vec_phi=diag_phi,
                                    posterior_inference_method=self._posterior_inference_method)

    def posterior(self, mat_obs: matrix, sample_weights: Optional[vector] = None, **kwargs) -> "NormalInverseWishart":
        if mat_obs.ndim == 1:
            mat_obs = mat_obs.reshape((1, -1))
        n_obs, n_dim = mat_obs.shape
        assert n_dim == self._n_dim, f"dimensionality mismatch: {n_dim} != {self._n_dim}"
        if sample_weights is not None:
            assert n_obs == len(sample_weights), f"sample size mismatch: {n_obs} != {len(sample_weights)}"
            # normalize sample weights as the sum equals to 1.0
            sample_weights = np.array(sample_weights) / np.sum(sample_weights)

        if self._posterior_inference_method == "default":
            return self._posterior_default(mat_obs=mat_obs, sample_weights=sample_weights)
        elif self._posterior_inference_method == "known_dof":
            # NIW with known \kappa and \nu. if not given externally, use ourselves.
            # This isn't recommended because covariance will diverge when \kappa or \nu - DoF is small compared to 1.0
            _kappa = kwargs.get("kappa_dash", self._kappa)
            _nu = kwargs.get("nu_dash", self._nu)
            return self._posterior_default(mat_obs=mat_obs, sample_weights=sample_weights, kappa_dash=_kappa, nu_dash=_nu)
        elif self._posterior_inference_method == "known_variance":
            return self._posterior_known_variance(mat_obs=mat_obs, sample_weights=sample_weights)
        elif self._posterior_inference_method in ("predictive_posterior", "mean_posterior"):
            if self._posterior_inference_method == "predictive_posterior":
                # predictive_posterior: returns NIW() with its mean and variance are identical to the "default" posterior predictive.
                predictive_posterior = self._posterior_default(mat_obs=mat_obs, sample_weights=sample_weights).approx_posterior_predictive()
                assert predictive_posterior.is_cov_diag, f"posterior predictive covariance is not diagonal."
                vec_mu = predictive_posterior.mean
                vec_cov_diag = np.diag(predictive_posterior.covariance)
            elif self._posterior_inference_method == "mean_posterior":
                vec_mu, vec_cov_diag = self._posterior_default(mat_obs=mat_obs, sample_weights=sample_weights).mean

            nu_minus_dof = self._nu - n_dim - 1
            # E[\Sigma] = \Phi / (\nu - p - 1) where p is the dimension size.
            vec_phi_diag = vec_cov_diag * nu_minus_dof
            posterior = NormalInverseWishart(vec_mu=vec_mu, kappa=self._kappa, nu=self._nu, vec_phi=vec_phi_diag,
                                             posterior_inference_method=self._posterior_inference_method)
            return posterior

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

    @property
    def mean(self) -> Tuple[np.ndarray, np.ndarray]:
        # it returns the expectation of the mean vector \mu and diagonal elements of covariance matrix \Sigma.
        # E[\mu] = \mu_{0}
        vec_mu = self._mu
        # E[\Sigma] = \frac{\Phi}{\nu - p - 1} for \nu > p + 1
        diag_cov_coef = 1.0 / (self._nu - self.n_dim - 1)
        vec_diag_cov = diag_cov_coef * self._diag_phi

        # return as tuple
        return (vec_mu, vec_diag_cov)