#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io
import warnings
from typing import Optional, Union, List, Any, Tuple
import pickle
import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal, norm
from scipy.special import iv, eval_genlaguerre
from scipy.spatial.distance import mahalanobis
from scipy.special import logsumexp
from scipy import optimize
from matplotlib import pyplot as plt

vector = np.array
matrix = np.ndarray
tensor = np.ndarray

from .mixture import MultiVariateGaussianMixture, _mvn_isotropic_logpdf, _mvn_isotropic_mahalanobis_dist_sq


class MultiVariateNormal(object):

    __EPS = 1E-5

    def __init__(self, vec_mu: vector,
                 mat_cov: Optional[matrix] = None, vec_cov: Optional[vector] = None, scalar_cov: Optional[float] = None):
        self._n_dim = len(vec_mu)
        self._mu = vec_mu
        if mat_cov is not None:
            self._cov = mat_cov
            self._is_cov_diag = False
            self._is_cov_iso = False
        elif vec_cov is not None:
            self._cov = np.diag(vec_cov)
            self._is_cov_diag = True
            self._is_cov_iso = False
        elif scalar_cov is not None:
            self._cov = np.eye(self._n_dim)*scalar_cov
            self._is_cov_diag = True
            self._is_cov_iso = True
        else:
            raise AttributeError("either `mat_cov` or `vec_cov` or `scalar_cov` must be specified.")
        self._validate()

    def _validate(self):
        msg = "dimensionality mismatch."
        assert len(self._mu) == self._n_dim, msg
        assert self._cov.shape[0] == self._n_dim, msg
        assert self._cov.shape[1] == self._n_dim, msg

        return True

    @property
    def n_dim(self):
        return self._n_dim

    @property
    def is_cov_diag(self):
        return self._is_cov_diag

    @property
    def is_cov_iso(self):
        return self._is_cov_iso

    @property
    def mean(self) -> np.ndarray:
        return self._mu

    @property
    def covariance(self) -> np.ndarray:
        return self._cov

    @property
    def entropy(self) -> float:
        """
        H[p(x)] = 0.5*ln(det(\Sigma)) + 0.5*D(1+ln(2\pi))
        :return: entropy of the multivariate normal distribution
        """
        e = multivariate_normal(mean=self._mu, cov=self._cov).entropy()
        return e

    @property
    def log_normalization_term(self) -> float:
        """
        ln(Z) = -0.5*ln(det(\Sigma)) - 0.5*D(ln(2\pi))
            = - (H[p(x)] - 0.5 * D)
        :return: log-normalization term of the multivariate normal distribution
        """
        z = - (self.entropy - 0.5 * self._n_dim)
        return z

    @property
    def l2_norm_mean(self) -> float:
        """
        expected value of the L2 norm of the vector that follows multivariate normal distribution.
        analytical solution utilizes the expected value of the non-central chi distribution.
        https://en.wikipedia.org/wiki/Noncentral_chi_distribution
        this method supports either isotropic or diagonal covariance matrix.
        for diagonal covariance, currently it applies equal variance approximation.

        * $E[||X||] = \sigma \sqrt{\pi/2} L^{d/2-1}_{1/2}(\lambda)$
            * $\lambda = \sqrt{(\sum_i{(\mu_i/\sigma)^2}}$
            * $L$: generalized laguerre polynomial
        :return: expected value of the L2-norm
        """

        if self.is_cov_iso:
            std = np.sqrt(self._cov[0][0])
        elif self.is_cov_diag:
            warnings.warn("approximates diagonal covariance with geometric mean.")
            std = np.exp(0.5*np.mean(np.log(self._cov.diagonal())))
        else:
            raise NotImplementedError("it supports diagonal covariance only.")

        if hasattr(self, "_l2_norm_mean"):
            return self._l2_norm_mean

        p_lambda = np.sqrt(np.sum((self._mu/std)**2))
        p_k = self.n_dim
        # generalized Laguerre polynomial; L^{(a)}_n(z)
        p_a = 0.5*p_k - 1
        p_n = 0.5
        p_z = - 0.5 * p_lambda**2
        l2_norm = std*np.sqrt(np.pi*0.5)*eval_genlaguerre(p_n, p_a, p_z)

        self._l2_norm_mean = l2_norm

        return l2_norm

    @property
    def l2_norm_variance(self) -> float:
        """
        covariance of the L2 norm of the vector that follows multivariate normal distribution.
        analytical solution exists for either isotropic or diagonal covariance matrix.
        """
        if hasattr(self, "_l2_norm_var"):
            return self._l2_norm_var

        l2_norm_mean = self.l2_norm_mean
        vec_cov = self._cov.diagonal()
        ret = np.sum(self._mu**2) + np.sum(vec_cov) - l2_norm_mean**2

        self._l2_norm_var = ret

        return ret

    @classmethod
    def random_generation(cls, n_dim: int, covariance_type="diagonal", mu_range=None, cov_range=None):
        lst_available_covariance_type = "identity,isotropic,diagonal,full".split(",")
        msg = "argument `covariance_type must be one of those: %s" % "/".join(lst_available_covariance_type)
        assert covariance_type in lst_available_covariance_type, msg

        rng_mu = [-2, 2] if mu_range is None else mu_range
        rng_cov = [0.2, 0.5] if cov_range is None else cov_range

        vec_mu = np.random.uniform(low=rng_mu[0], high=rng_mu[1], size=n_dim)
        if covariance_type == "isotropic":
            scalar_cov = np.random.uniform(low=rng_cov[0], high=rng_cov[1], size=1)
            ret = cls(vec_mu, scalar_cov=scalar_cov)
        elif covariance_type == "diagonal":
            vec_cov = np.random.uniform(low=rng_cov[0], high=rng_cov[1], size=n_dim)
            ret = cls(vec_mu, vec_cov=vec_cov)
        elif covariance_type == "identity":
            scalar_cov = 1.
            ret = cls(vec_mu, scalar_cov=scalar_cov)
        elif covariance_type == "full":
            # sample from inverse-wishart distribution.
            scale_param = np.diag(np.random.uniform(low=rng_cov[0], high=rng_cov[1], size=n_dim))
            dof = n_dim + 2
            mat_cov = sp.stats.invwishart.rvs(df=dof, scale=scale_param)
            ret = cls(vec_mu, mat_cov=mat_cov)
        else:
            raise NotImplementedError("unexpected input.")

        return ret

    @classmethod
    def tuple_to_tensor(cls, lst_of_tuple):
        mu = np.stack(tup[0] for tup in lst_of_tuple)
        cov = np.stack(tup[1] for tup in lst_of_tuple)

        return mu, cov

    @classmethod
    def tensor_to_tuple(cls, mat_mu: matrix, tensor_cov: tensor):
        lst_ret = []
        n_k = mat_mu.shape[0]
        for k in range(n_k):
            lst_ret.append((mat_mu[k], tensor_cov[k]))

        return lst_ret

    @classmethod
    def mvn_to_gmm(cls, lst_distribution: List["MultiVariateNormal"],
                   lst_weight: Optional[Union[List[float], np.ndarray]] = None) -> "MultiVariateGaussianMixture":
        """
        concatenate multiple normal distributions into single gaussian mixture distribution.

        :param lst_distribution: list of gaussian mixture instances.
        :param lst_weight: list of relative weights that are applied to each instance.
        """
        n = len(lst_distribution)
        if lst_weight is None:
            lst_weight = np.full(n, fill_value=1./n)
        else:
            if isinstance(lst_weight, list):
                lst_weight = np.array(lst_weight)
            assert len(lst_weight) == len(lst_distribution), "length mismatch detected."
            assert np.abs(np.sum(lst_weight) - 1.) < cls.__EPS, "sum of relative weight must be equal to 1."
        # sanity check
        n_dim = lst_distribution[0].n_dim
        assert all([dist.n_dim == n_dim for dist in lst_distribution]), "dimension size mismatch detected."

        # concatenate gaussian mixture parameters
        vec_alpha = lst_weight
        mat_mu = np.vstack([dist.mean for dist in lst_distribution])
        if all([dist.is_cov_iso for dist in lst_distribution]):
            vec_cov = np.array([dist.covariance[0][0] for dist in lst_distribution])
            dist_new = MultiVariateGaussianMixture(vec_alpha=vec_alpha, mat_mu=mat_mu, vec_std=np.sqrt(vec_cov))
        elif all([dist.is_cov_diag for dist in lst_distribution]):
            mat_cov = np.vstack([np.diag(dist.covariance) for dist in lst_distribution])
            dist_new = MultiVariateGaussianMixture(vec_alpha=vec_alpha, mat_mu=mat_mu, mat_cov=mat_cov)
        else:
            # tensor_cov = [n_component, n_dim, n_dim]
            tensor_cov = np.stack([dist.covariance for dist in lst_distribution])
            dist_new = MultiVariateGaussianMixture(vec_alpha=vec_alpha, mat_mu=mat_mu, tensor_cov=tensor_cov)

        return dist_new

    def save(self, file_path: str):
        assert self._validate(), "corrupted inner structure detected."
        with io.open(file_path, mode="wb") as ofs:
            pickle.dump(self, ofs)

    @classmethod
    def load(cls, file_path: str) -> "MultiVariateNormal":
        with io.open(file_path, mode="rb") as ifs:
            obj = pickle.load(ifs)
        obj.__class__ = cls
        return obj

    def density_plot(self, fig_and_ax=None, vis_range: Optional[Tuple[float]] = None,
                     figsize: Optional[Tuple[float]] = None,
                     annotation: Optional[str] = None, n_mesh_bin: int = 100, cmap: str = "Reds", **kwargs):
        assert self._n_dim == 2, "visualization isn't available except 2-dimensional distribution."

        rng_default = np.max(np.abs(self._mu)) + 2. * np.sqrt(np.max(self._cov))
        rng = [-rng_default, rng_default] if vis_range is None else vis_range

        mesh_x, mesh_y = np.meshgrid(np.linspace(rng[0], rng[1], n_mesh_bin), np.linspace(rng[0], rng[1], n_mesh_bin))
        mesh_xy = np.vstack([mesh_x.flatten(), mesh_y.flatten()])

        if fig_and_ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = fig_and_ax[0], fig_and_ax[1]

        value_z = self.pdf(mesh_xy.T)

        ax.pcolormesh(mesh_x, mesh_y, value_z.reshape(mesh_x.shape), cmap=cmap, **kwargs)

        if annotation is not None:
            ax.annotate(annotation, (self._mu[0], self._mu[1]))

        return fig, ax

    def pdf(self, vec_x: Union[vector, matrix]) -> np.ndarray:
        if vec_x.ndim == 1:
            prob = multivariate_normal.pdf(vec_x, self._mu, self._cov)
        else:
            prob = np.array([multivariate_normal.pdf(x, self._mu, self._cov) for x in vec_x])
        return prob

    def logpdf(self, vec_x: Union[vector, matrix]) -> np.ndarray:
        if vec_x.ndim == 1:
            if self._is_cov_diag:
                ln_prob = _mvn_isotropic_logpdf(vec_x, self._mu, self._cov)
            else:
                ln_prob = multivariate_normal.logpdf(vec_x, self._mu, self._cov)
        elif vec_x.ndim == 2:
            if self._is_cov_diag:
                ln_prob = np.array([_mvn_isotropic_logpdf(x, self._mu, self._cov) for x in vec_x])
            else:
                ln_prob = np.array([multivariate_normal.logpdf(x, self._mu, self._cov) for x in vec_x])
        else:
            raise NotImplementedError("unexpected input.")

        return ln_prob

    def mahalanobis_distance_sq(self, vec_x: Union[vector, matrix]):
        """
        returns the square of the mahalanobis distance using mean and covariance of the distribution.
        d = (x - \mu)^T \Sigma^{-1} (x - \mu)
        -0.5*mahalanobis_distance_sq + log_normalization_term = logpdf

        :param vec_x: observation(s)
        :return: (array of) square of the mahalanobis distance.
        """
        if vec_x.ndim == 1:
            if self._is_cov_diag:
                dist = _mvn_isotropic_mahalanobis_dist_sq(vec_x, self._mu, self._cov)
            else:
                cov_inv = np.linalg.inv(self._cov)
                dist = mahalanobis(vec_x, self._mu, cov_inv)**2
        elif vec_x.ndim == 2:
            if self._is_cov_diag:
                dist = np.array([_mvn_isotropic_mahalanobis_dist_sq(x, self._mu, self._cov) for x in vec_x])
            else:
                cov_inv = np.linalg.inv(self._cov)
                dist = np.array([mahalanobis(x, self._mu, cov_inv)**2 for x in vec_x])
        else:
            raise NotImplementedError("unexpected input.")

        return dist

    def cosine_similarity(self, vec_x: vector):
        """
        returns the cosine similarity between the input and mean vector
        :param vec_x: observation
        :return: cos(vec_x, \mu)
        """
        vec_y = self._mu
        cosine = np.sum(vec_x*vec_y) / (np.linalg.norm(vec_x)*np.linalg.norm(vec_y))

        return cosine

    def random(self, size: int):
        """
        generate random samples from the distribution.

        :param size: number of samples
        :return: generated samples
        """
        mat_r_x = np.random.multivariate_normal(mean=self._mu, cov=self._cov, size=size, check_valid="ignore")
        return mat_r_x

    def radon_transform(self, vec_theta: vector):
        mu_t = self._mu.dot(vec_theta) # mu[k]^T.theta
        std_t = np.sqrt(vec_theta.dot(self._cov).dot(vec_theta)) # theta^T.cov[k].theta
        return sp.stats.norm(mu_t, std_t)

    def inter_distance(self, p_y: "MultiVariateNormal", metric: str) -> np.ndarray:
        """
        returns distance between two normal distributions.
        it supports normal distribution with diagonal covariance matrix only.
        supported metrics are: squared 2-wasserstein(wd_sq), kullback-leibler(kl), jensen-shannon(js), expected likelihood kernel(elk)

        :param metric:
        """
        assert self.is_cov_diag, "currently it supports diagonal covariance only."
        assert p_y.is_cov_diag, "currently it supports diagonal covariance only."

        # mat_mu_x = self._mu
        # mat_cov_x = np.stack([np.diag(cov) for cov in self._cov])
        # mat_mu_y = p_y._mu
        # mat_cov_y = np.stack([np.diag(cov) for cov in p_y._cov])
        #
        # if metric == "wd_sq":
        #     mat_std_x, mat_std_y = np.sqrt(mat_cov_x), np.sqrt(mat_cov_y)
        #     mat_dist = _wasserstein_distance_sq_between_multivariate_normal_diag_parallel(
        #                 mat_mu_x=mat_mu_x, mat_std_x=mat_std_x, mat_mu_y=mat_mu_y, mat_std_y=mat_std_y)
        # elif metric == "kl":
        #     mat_dist = _kldiv_diag_parallel(mat_mu_x=mat_mu_x, mat_cov_x=mat_cov_x, mat_mu_y=mat_mu_y, mat_cov_y=mat_cov_y)
        # elif metric in ["js","js_an"]:
        #     mat_dist_xy = _kldiv_diag_parallel(mat_mu_x=mat_mu_x, mat_cov_x=mat_cov_x, mat_mu_y=mat_mu_y, mat_cov_y=mat_cov_y)
        #     mat_dist_yx = _kldiv_diag_parallel(mat_mu_x=mat_mu_y, mat_cov_x=mat_cov_y, mat_mu_y=mat_mu_x, mat_cov_y=mat_cov_x)
        #     mat_dist = 0.5*(mat_dist_xy + mat_dist_yx.T)
        # elif metric == "elk":
        #     mat_sim_xy = _expected_likelihood_kernel_multivariate_normal_diag_parallel(mat_mu_x=mat_mu_x, mat_cov_x=mat_cov_x, mat_mu_y=mat_mu_y, mat_cov_y=mat_cov_y)
        #     mat_dist = np.log(1. - np.exp(mat_sim_xy))
        # elif metric == "elk_log_neg":
        #     mat_sim_xy = _expected_likelihood_kernel_multivariate_normal_diag_parallel(mat_mu_x=mat_mu_x, mat_cov_x=mat_cov_x, mat_mu_y=mat_mu_y, mat_cov_y=mat_cov_y)
        #     mat_dist = -mat_sim_xy
        # else:
        #     raise NotImplementedError(f"unsupported metric was specified: {metric}")
        #
        # return mat_dist

    def _calc_principal_component_vectors(self, n_dim: int):
        assert n_dim < self.n_dim, "reduced dimension size `n_dim` must be smaller than original dimension size."

        # covariance matrix
        mat_cov = self.covariance
        # eigen decomposition
        vec_l, mat_w = np.linalg.eig(mat_cov)
        # take largest top-k eigenvectors
        idx_rank = vec_l.argsort()[::-1]
        mat_w_h = mat_w[:, idx_rank[:n_dim]]
        # returned matrix shape will be (self.n_dim, n_dim). each column is a i-th eigenvector
        return mat_w_h

    def dimensionality_reduction_by_pca(self, n_dim: int, factor_loading_matrix: Optional[np.ndarray] = None):

        # calculate factor loading matrix; (n_dim, n_dim_r)
        if factor_loading_matrix is not None:
            mat_w_h = factor_loading_matrix
            assert mat_w_h.shape[1] == n_dim, "specified factor loading matrix is inconsistent with `n_dim` argument."
        else:
            mat_w_h = self._calc_principal_component_vectors(n_dim)
        # transform mean vector; (n_dim_r,)
        vec_mu_h = self._mu.dot(mat_w_h)
        # transform covariance matrix; (n_component, n_dim_r, n_dim_r)
        mat_cov_h = mat_w_h.T.dot(self._cov).dot(mat_w_h)
        # create new instance
        ret = MultiVariateNormal(vec_mu=vec_mu_h, mat_cov=mat_cov_h)

        return ret

    def normalize(self, inplace=False):

        scalar_scale = np.linalg.norm(self._mu)
        mu_norm = self._mu / scalar_scale
        cov_norm = self._cov / scalar_scale**2

        if inplace:
            self._mu = mu_norm
            self._cov = cov_norm
        else:
            if self.is_cov_iso:
                scalar_cov = cov_norm[0,0]
                return MultiVariateNormal(vec_mu=mu_norm, scalar_cov=scalar_cov)
            elif self.is_cov_diag:
                vec_cov = np.diag(cov_norm)
                return MultiVariateNormal(vec_mu=mu_norm, vec_cov=vec_cov)
            else:
                return MultiVariateNormal(vec_mu=mu_norm, vec_cov=cov_norm)


# utility functions
def _hiv(alpha, x):
    if x == 0.:
        return 0.
    return iv(alpha, x) / iv(alpha-1, x)

class vonMisesFisher(object):

    __EPS = 1E-5

    def __init__(self, vec_mu: vector, scalar_kappa: float):
        self._n_dim = len(vec_mu)
        self._mu = vec_mu
        self._kappa = scalar_kappa
        self._validate()
        self._normalization_term = self.calc_normalization_term(self._n_dim, self._kappa)

    def _validate(self):
        assert np.abs(1. - np.linalg.norm(self._mu)) < self.__EPS, "`vec_mu` must be unit vector."
        assert self._kappa >= 0, "`scalar_kappa` must be positive."
        return True

    @property
    def n_dim(self):
        return self._n_dim

    @property
    def mu(self) -> np.ndarray:
        return self._mu

    @property
    def kappa(self) -> float:
        return self._kappa

    @property
    def mean(self) -> np.ndarray:
        coef = _hiv(0.5*self._n_dim, self._kappa)
        return self._mu * coef

    @property
    def covariance(self) -> np.ndarray:
        cov_mu = np.outer(self._mu, self._mu)
        cov_eye = np.eye(self._n_dim, dtype=np.float)
        h = _hiv(0.5*self._n_dim, self._kappa)
        cov = (h/self._kappa)*cov_eye + (1 - self._n_dim*h/self._kappa - h**2)*cov_mu
        return cov

    @property
    def entropy(self) -> float:
        # H[p] = c_p(\kappa) - \kappa<\mu, E[x]> = c_p(\kappa) - \kappa * hiv(n_dim/2,\kappa)
        e = self.log_normalization_term - self._kappa * _hiv(alpha=self._n_dim*0.5, x=self._kappa)
        return e

    @property
    def normalization_term(self) -> float:
        return self._normalization_term

    @property
    def log_normalization_term(self) -> float:
        return np.log(self._normalization_term)

    @classmethod
    def calc_normalization_term(cls, n_dim, kappa) -> float:
        d_dash = 0.5*n_dim - 1
        norm = kappa**d_dash
        denom = np.power(2*np.pi, 0.5*n_dim) * iv(d_dash, kappa)

        return 1./ (norm / denom) # 1 / C_p(\kappa)

    @classmethod
    def random_generation(cls, n_dim: int, mu_range=None, kappa_range=None):
        rng_mu = [-2, 2] if mu_range is None else mu_range
        rng_kappa = [0.1, 3.0] if kappa_range is None else kappa_range

        vec_mu = np.random.uniform(low=rng_mu[0], high=rng_mu[1], size=n_dim)
        vec_mu /= np.linalg.norm(vec_mu)
        scalar_kappa = np.random.uniform(low=rng_kappa[0], high=rng_kappa[1], size=1)[0]
        ret = cls(vec_mu, scalar_kappa)

        return ret

    def save(self, file_path: str):
        assert self._validate(), "corrupted inner structure detected."
        with io.open(file_path, mode="wb") as ofs:
            pickle.dump(self, ofs)

    @classmethod
    def load(cls, file_path: str) -> "vonMisesFisher":
        with io.open(file_path, mode="rb") as ifs:
            obj = pickle.load(ifs)
        obj.__class__ = cls
        return obj

    def density_plot(self, fig_and_ax=None, vis_range: Optional[Tuple[float]] = None,
                     figsize: Optional[Tuple[float]] = None,
                     annotation: Optional[str] = None, n_mesh_bin: int = 100, cmap: str = "Reds", **kwargs):
        assert self._n_dim == 2, "visualization isn't available except 2-dimensional distribution."

        """
        Ideally I want to visualize it on the surface of the sphere (3D) or on the edge of the circle (2D),
        but it exceeds my knowledge.
        Instead, I regard this as the 2-dimensional multivariate normal distribution for the time being.
        """
        p_mvn = MultiVariateNormal(vec_mu=self._mu, mat_cov=self.covariance)
        fig, ax = p_mvn.density_plot(fig_and_ax, vis_range, figsize, annotation, n_mesh_bin, cmap, **kwargs)
        return fig, ax

    def _likelihood(self, x: np.ndarray):
        return np.exp(self._kappa*np.dot(x, self._mu))

    def _log_likelihood(self, x):
        return self._kappa*np.dot(x, self._mu)

    def _sample_ortho_norm_vector(self, theta: np.ndarray, size: int):
        n_dim = len(theta)
        x = np.random.rand(n_dim*size).reshape(size,n_dim) - 0.5
        # orthogonal to theta
        # x = x - <x,\theta> / \theta / n_dim
        x = x - np.dot(x, theta).reshape(size,1) * (1./(n_dim*theta)).reshape(1,n_dim)
        # normalize
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        return x

    def _sample_weight_batch(self, vec_kappa: np.ndarray, n_dim: int, size: int):
        result = np.stack([self._sample_weight(kappa, n_dim, size) for kappa in vec_kappa])
        return result

    def _sample_weight(self, kappa, n_dim: int, size: int):
        """
        Rejection sampling scheme for sampling distance from center on surface of the sphere.
        borrowed from: https://github.com/jiacheng-xu/vmf_vae_nlp
        """
        dim = n_dim - 1  # since S^{n-1}
        b = dim / (np.sqrt(4. * kappa ** 2 + dim ** 2) + 2 * kappa)
        x = (1. - b) / (1. + b)
        c = kappa * x + dim * np.log(1 - x ** 2)

        sampled = 0
        vec_result = np.zeros(shape=(size,), dtype=np.float)
        while True:
            z = np.random.beta(dim / 2., dim / 2.)  # concentrates towards 0.5 as d-> inf
            w = (1. - (1. + b) * z) / (1. - (1. - b) * z)
            u = np.random.uniform(low=0, high=1)
            if kappa * w + dim * np.log(1. - x * w) - c >= np.log(u):
                vec_result[sampled] = w
                sampled += 1
                if sampled == size:
                    return vec_result

    def pdf(self, vec_x: Union[vector, matrix]) -> np.ndarray:
        l = self._likelihood(vec_x)
        norm = self.normalization_term
        return l / norm

    def logpdf(self, vec_x: Union[vector, matrix]) -> np.ndarray:
        ln_l = self._log_likelihood(vec_x)
        ln_norm = self.log_normalization_term
        return ln_l - ln_norm

    def random(self, size: int):
        """
        generate random samples from the distribution.

        :param size: number of samples
        :return: generated samples
        """
        n_dim = self._n_dim
        vec_w = self._sample_weight(self._kappa, n_dim, size).reshape(size, 1)
        mat_v = self._sample_ortho_norm_vector(self._mu, size)
        # x = sqrt(1-w**2)*v+w*mu
        mat_result = mat_v * np.sqrt(1.-vec_w**2) + vec_w*self._mu.reshape(1,n_dim)

        if size == 1:
            mat_result = mat_result.flatten()

        return mat_result

    def inter_distance(self, p_y: "MultiVariateNormal", metric: str) -> float:
        """
        returns distance between two normal distributions.
        it supports normal distribution with diagonal covariance matrix only.
        supported metrics are: squared 2-wasserstein(wd_sq), kullback-leibler(kl), jensen-shannon(js), expected likelihood kernel(elk)

        :param metric:
        """
        raise NotImplementedError("not implemented yet.")

    def _calc_principal_component_vectors(self, n_dim: int):
        assert n_dim < self.n_dim, "reduced dimension size `n_dim` must be smaller than original dimension size."

        # covariance matrix
        mat_cov = self.covariance
        # eigen decomposition
        vec_l, mat_w = np.linalg.eig(mat_cov)
        # take largest top-k eigenvectors
        idx_rank = vec_l.argsort()[::-1]
        mat_w_h = mat_w[:, idx_rank[:n_dim]]
        # returned matrix shape will be (self.n_dim, n_dim). each column is a i-th eigenvector
        return mat_w_h

    def _find_root_kappa_for_dimensionality_reduction(self, n_dim_target) -> float:
        objective_value = _hiv(alpha=self._n_dim*0.5, x=self._kappa)
        alpha = n_dim_target*0.5
        def objective_function(kappa):
            return _hiv(alpha=alpha, x=kappa) - objective_value

        rng_kappa = (0., self._kappa)
        kappa_new = sp.optimize.bisect(objective_function, *rng_kappa)

        return kappa_new

    def dimensionality_reduction_by_pca(self, n_dim: int, factor_loading_matrix: Optional[np.ndarray] = None):

        # it seems that naive linear transformation doesn't work as expected.
        # I have to nothing but disable this method until I come up with a good solution.
        raise NotImplementedError("you can't use this method for the time being.")

        assert 1 < n_dim < self._n_dim, f"`n_dim` must be smaller than original dimension size: {self._n_dim}"

        # calculate factor loading matrix; (n_dim, n_dim_r)
        if factor_loading_matrix is not None:
            mat_w_h = factor_loading_matrix
            assert mat_w_h.shape[1] == n_dim, "specified factor loading matrix is inconsistent with `n_dim` argument."
        else:
            mat_w_h = self._calc_principal_component_vectors(n_dim)
        # transform direction vector; (n_dim_r,)
        vec_mu_h = self._mu.dot(mat_w_h)
        vec_mu_h /= np.linalg.norm(vec_mu_h)
        # calculate best scale parameter that preserves
        kappa_h = self._find_root_kappa_for_dimensionality_reduction(n_dim_target=n_dim)
        # create new instance
        ret = vonMisesFisher(vec_mu_h, kappa_h)

        return ret