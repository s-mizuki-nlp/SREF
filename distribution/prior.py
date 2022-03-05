#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os, sys, io
import warnings
from typing import Optional, Union, List, Any, Tuple, Dict
import pickle

import mpmath
import numpy as np
import scipy as sp
from scipy import optimize
from scipy.special import logsumexp

vector = np.array
matrix = np.ndarray
tensor = np.ndarray

from .continuous import MultiVariateNormal, vonMisesFisher, _hiv

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
                           kappa_dash: Optional[float] = None, nu_dash: Optional[float] = None,
                           normalize: bool = False) -> "NormalInverseWishart":

        n_obs, n_dim = mat_obs.shape

        # prior mean
        vec_mu_0 = self._mu

        # empirical mean
        if sample_weights is None:
            vec_mu_e = mat_obs.mean(axis=0)
        else:
            # E[X] = \sum_{i}p(x_i)x_i
            vec_mu_e = np.sum(mat_obs * sample_weights.reshape(n_obs,1), axis=0)

        if normalize:
            vec_mu_0 = vec_mu_0 / (np.linalg.norm(vec_mu_0) + 1E-15)
            vec_mu_e = vec_mu_e / (np.linalg.norm(vec_mu_e) + 1E-15)

        # empirical covariance
        if n_obs == 1:
            diag_moment = np.zeros(shape=(n_dim,), dtype=np.float)
        else:
            # diag_moment = \sum_{i=1}^{n}(x_{ik}-\bar{x_{k}})^2
            if sample_weights is None:
                diag_moment = np.sum((mat_obs - vec_mu_e)**2, axis=0)
            else:
                diag_moment = np.sum( ((mat_obs - vec_mu_e)**2) * n_obs * sample_weights.reshape(n_obs,1), axis=0)

        # \mu: weighted avg between \mu_0 and \mu_e
        vec_mu_dash = (self._kappa * vec_mu_0 + n_obs * vec_mu_e) / (self._kappa + n_obs)
        if normalize:
            vec_mu_dash = vec_mu_dash / (np.linalg.norm(vec_mu_dash) + 1E-15)

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
        diag_mu_diff_moment  = (vec_mu_e - vec_mu_0)**2
        diag_mu_diff_coef = (self._kappa * n_obs) / (self._kappa + n_obs)
        diag_phi = self._diag_phi + diag_moment + diag_mu_diff_coef * diag_mu_diff_moment

        return NormalInverseWishart(vec_mu=vec_mu_dash, kappa=_kappa, nu=_nu, vec_phi=diag_phi,
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


class vonMisesFisherConjugatePrior(object):

    __EPS = 1E-5
    _AVAILABLE_POSTERIOR_INFERENCE_METHOD = ["default", "known_dof"]

    def __init__(self, vec_mu: vector, r_0: float, c: float,
                 posterior_inference_method: str):

        assert posterior_inference_method in self._AVAILABLE_POSTERIOR_INFERENCE_METHOD, \
            f"invalid `posterior_inference_method` value. valid values are: {self._AVAILABLE_POSTERIOR_INFERENCE_METHOD}"

        self._n_dim = len(vec_mu)
        self._posterior_inference_method = posterior_inference_method
        self._mu = vec_mu
        self._r_0 = r_0
        self._c = c

    @classmethod
    def AVAILABLE_POSTERIOR_INFERENCE_METHOD(cls):
        return cls._AVAILABLE_POSTERIOR_INFERENCE_METHOD

    def mean(self, n_estimation: int = int(1E4)) -> Tuple[float, np.ndarray]:
        # approximate expectation of \mu and \kappa using the algorithm proposed in [Gabriel and Eduardo, 2005].
        # for initial value of \mu and \kappa, we use MAP estimator instead of Maximum Likelihood estimator which was employed in original work.
        kappa_map, vec_mu_map = self.map()
        kappa_e, vec_mu_e = self._approximate_mean_sir_method(mu_mle=vec_mu_map, kappa_mle=kappa_map, kappa_variance=100.0,
                                                              m_0=self._mu, r_0=self._r_0, c=self._c, p=self._n_dim, q_c=0.5, n_samples=n_estimation)
        return kappa_e, vec_mu_e

    def map(self) -> Tuple[float, np.ndarray]:
        # return approximated MAP estimator
        vec_mu = self._mu
        kappa = self._approximate_kappa_map(r_0=self._r_0, c=self._c, p=self._n_dim)

        return (kappa, vec_mu)

    def _approximate_kappa_map(self, r_0: float, c: float, p: Union[int, float]) -> float:
        """
        return MAP estimator of \kappa when \mu = \mu_{MAP}
        """
        target_value = r_0 / c
        # objective_function: grad_{\kappa} ln \pi(\kappa|\mu=\mu_{MAP}; c, R_0, m_0)
        # this function is monotone increasing to \kappa.
        def objective_function(kappa):
            return _hiv(alpha=p*0.5, x=kappa) - target_value

        k_min = 0.0
        k_max = 100.0
        while True:
            if objective_function(k_max) < 0:
                k_max = k_max * 2
            else:
                break
        value_range = (k_min, k_max)
        k_map = optimize.bisect(objective_function, *value_range)

        return k_map

    def _posterior_default(self, mat_obs: matrix, sample_weights: Optional[vector] = None) -> "vonMisesFisherConjugatePrior":
        n_obs, n_dim = mat_obs.shape
        if sample_weights is not None:
            # \bar{x} = N \sum_{i}{w_i x_i}
            vec_x_bar = np.sum(n_obs * sample_weights.reshape(-1,1) * mat_obs, axis=0)
        else:
            vec_x_bar = np.sum(mat_obs, axis=0)

        # formula [2.2] in [Gabriel and Eduardo, 2005]
        # \hat{\mu} = R_0 \mu_0 + \bar{x}
        vec_mu_hat = self._r_0 * self._mu + vec_x_bar
        # R_n = ||\hat{\mu}||
        r_n = np.linalg.norm(vec_mu_hat)
        # m_n = \hat{\mu} / R_n
        vec_m_n = vec_mu_hat / r_n
        # c_n = c + n
        c_n = self._c + n_obs
        return vonMisesFisherConjugatePrior(vec_mu=vec_m_n, r_0=r_n, c=c_n, posterior_inference_method=self._posterior_inference_method)

    def posterior(self, mat_obs: matrix, sample_weights: Optional[vector] = None, **kwargs) -> "vonMisesFisherConjugatePrior":
        if mat_obs.ndim == 1:
            mat_obs = mat_obs.reshape((1, -1))
        n_obs, n_dim = mat_obs.shape
        assert n_dim == self._n_dim, f"dimensionality mismatch: {n_dim} != {self._n_dim}"
        if sample_weights is not None:
            assert n_obs == len(sample_weights), f"sample size mismatch: {n_obs} != {len(sample_weights)}"
            # normalize sample weights as the sum equals to 1.0
            sample_weights = np.array(sample_weights) / np.sum(sample_weights)

        if self._posterior_inference_method == "default":
            # it returns posterior distribution
            return self._posterior_default(mat_obs=mat_obs, sample_weights=sample_weights)

        elif self._posterior_inference_method == "known_dof":
            # inhrerit own {c, R_0} values to the posterior unless explicitly specified.
            c_dash = kwargs.get("c", self._c)
            r_0_dash = kwargs.get("r_0", self._r_0)
            p_post_temp = self._posterior_default(mat_obs=mat_obs, sample_weights=sample_weights)
            return vonMisesFisherConjugatePrior(vec_mu=p_post_temp._mu, r_0=r_0_dash, c=c_dash, posterior_inference_method=self._posterior_inference_method)

    def _approximate_mean_sir_method(self, mu_mle: vector, kappa_mle: float, kappa_variance: float = 100.0,
                                     m_0: Optional[vector] = None, r_0: Optional[float] = None, c: Optional[float] = None, p: Optional[float] = None,
                                     q_c: float = 0.5,
                                     n_samples: int = int(1E4)) -> Tuple[float, np.ndarray]:
        # estimate posterior mean of \mu and \kappa using sampling-importance-resampling (SIR) algorithm.
        # ref: [Gabriel and Eduardo, 2005] A Bayesian Analysis of Directional Data using the von Mises-Fisher Distribution

        @np.vectorize
        def _log_norm(kappa, c, p):
            log_num_c_p = (p/2-1)*np.log(kappa)
            log_denom_c_p = float(mpmath.log(mpmath.besseli(p/2-1, kappa)))
            log_norm = c * (log_num_c_p - log_denom_c_p)
            return log_norm

        def _target_log_likelihood(mat_mu, vec_kappa, m_0, r_0, c, p):
            # r_0 = self._r_0, c = self._c, p = self._n_dim
            # mat_mu: sampled \mu s from proposal distribution
            # vec_kappa: sampled \kappa s from proposal distribution
            # log(exp): \kappa R_n <\mu, m_n>
            vec_log_exp = vec_kappa * r_0 * mat_mu.dot(m_0)
            vec_log_norm_inv = _log_norm(kappa=vec_kappa, c=c, p=p)
            llk = vec_log_norm_inv + vec_log_exp
            return llk

        # sampling from proposal distributions
        # \mu proposal: vMF(x; \hat{mu}, Q_c \hat{\kappa} R_n)
        r_0 = self._r_0 if r_0 is None else r_0
        vec_mu_samples = vonMisesFisher(vec_mu=mu_mle, scalar_kappa=q_c * kappa_mle * r_0).random(size=n_samples)
        # \kappa proposal: Gamma(k; mean, var)
        # mean = \hat{\kappa}, var = 100 or \hat{V[\kappa]}
        # configure np.random.gamma(shape, scale)
        k_shape = kappa_mle**2 / kappa_variance
        theta_scale = kappa_variance / kappa_mle
        vec_kappa_samples = np.random.gamma(shape=k_shape, scale=theta_scale, size=n_samples)

        # calculate sample weights using log-likelihood on target distribution (=oneself)
        m_0 = self._mu if m_0 is None else m_0
        c = self._c if c is None else c
        p = self._n_dim if p is None else p
        vec_tgt_llk = _target_log_likelihood(mat_mu=vec_mu_samples, vec_kappa=vec_kappa_samples, m_0=m_0, r_0=r_0, c=c, p=p)
        # calculate sample weights
        vec_weights = np.exp( vec_tgt_llk - logsumexp(vec_tgt_llk) )
        # calculate SIR expectation
        vec_mu_e = np.sum(vec_mu_samples * vec_weights.reshape(-1,1), axis=0)
        vec_mu_e = vec_mu_e / np.linalg.norm(vec_mu_e)
        kappa_e = np.sum(vec_weights * vec_kappa_samples)

        return kappa_e, vec_mu_e
