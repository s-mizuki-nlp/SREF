#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
from scipy.stats import multivariate_normal
from distribution.continuous import MultiVariateNormal


def _extract_params(p: "MultiVariateNormal"):
    vec_mu = p.mean
    if p.is_cov_diag:
        cov = np.diag(p.covariance)
    else:
        cov = p.covariance
    return vec_mu, cov

# kullback-leibler divergence
def kullback_leibler_mvn(p_x: "MultiVariateNormal", p_y: "MultiVariateNormal"):

    vec_mu_x, vm_cov_x = _extract_params(p_x)
    vec_mu_y, vm_cov_y = _extract_params(p_y)
    if p_x.is_cov_diag and p_y.is_cov_diag:
        kldiv = _kldiv_mvn_diag(vec_mu_x, vm_cov_x, vec_mu_y, vm_cov_y)
    else:
        kldiv = _kldiv_mvn_full(vec_mu_x, vm_cov_x, vec_mu_y, vm_cov_y)

    return kldiv

def _kldiv_mvn_diag(vec_mu_x: np.ndarray, vec_cov_x: np.ndarray, vec_mu_y: np.ndarray, vec_cov_y: np.ndarray):

    d = vec_mu_x.size
    ln_det_cov1 = np.sum(np.log(vec_cov_x))
    ln_det_cov2 = np.sum(np.log(vec_cov_y))
    tr_cov12 = np.sum((vec_cov_y/vec_cov_x))
    quad_12 = np.sum(((vec_mu_x-vec_mu_y)/np.sqrt(vec_cov_x))**2)

    kldiv = 0.5*(tr_cov12 + quad_12 - d - ln_det_cov2 + ln_det_cov1)
    return kldiv

def _kldiv_mvn_full(vec_mu_x: np.ndarray, mat_cov_x: np.ndarray, vec_mu_y: np.ndarray, mat_cov_y: np.ndarray):

    vec_cov_x = np.diag(mat_cov_x)
    vec_cov_y = np.diag(mat_cov_y)
    mat_prec_x = np.linalg.inv(mat_cov_x)
    vec_mu_xy = vec_mu_x-vec_mu_y

    d = vec_mu_x.size
    _, ln_det_cov1 = np.linalg.slogdet(mat_cov_x)
    _, ln_det_cov2 = np.linalg.slogdet(mat_cov_y)
    tr_cov12 = np.sum((vec_cov_y/vec_cov_x))
    quad_12 = vec_mu_xy.dot(mat_prec_x).dot(vec_mu_xy.T)

    kldiv = 0.5*(tr_cov12 + quad_12 - d - ln_det_cov2 + ln_det_cov1)
    return kldiv


def expected_likelihood_mvn(p_x: "MultiVariateNormal", p_y: "MultiVariateNormal", log: bool = True):

    vec_mu_x, vm_cov_x = _extract_params(p_x)
    vec_mu_y, vm_cov_y = _extract_params(p_y)

    n_dim = len(vec_mu_x)
    vec_mu_xy = vec_mu_x - vec_mu_y
    vm_cov_xy = vm_cov_x + vm_cov_y
    vec_x = np.zeros(n_dim, dtype=vec_mu_x.dtype)

    if log:
        elk = multivariate_normal.logpdf(vec_x, vec_mu_xy, vm_cov_xy)
    else:
        elk = multivariate_normal.pdf(vec_x, vec_mu_xy, vm_cov_xy)

    return elk
