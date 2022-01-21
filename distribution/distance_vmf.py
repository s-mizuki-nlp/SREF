#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import scipy as sp
from distribution.continuous import vonMisesFisher

def _extract_params(p: "vonMisesFisher"):
    vec_mu = p.mu
    kappa = p.kappa
    return vec_mu, kappa


def expected_likelihood_vmf(p_x: "vonMisesFisher", p_y: "vonMisesFisher", log: bool = True):

    n_dim = p_x.n_dim
    mu_x, k_x = _extract_params(p_x)
    mu_y, k_y = _extract_params(p_y)

    kappa_xy = np.sqrt(k_x**2+k_y**2+2*k_x*k_y*np.dot(mu_x,mu_y))
    norm_x = p_x.normalization_term
    norm_y = p_y.normalization_term
    norm_xy = vonMisesFisher.calc_normalization_term(n_dim, kappa_xy)

    # ELK = c_p(dim,\kappa_{x})*c_p(dim,\kappa_{y} / c_p(dim,\kappa_{xy})
    # c_p = 1 / norm
    if log:
        elk = np.log(norm_xy) - (np.log(norm_x) + np.log(norm_y))
    else:
        elk = norm_xy/(norm_x*norm_y)

    return elk


def kullback_leibler_vmf(p_x: "vonMisesFisher", p_y: "vonMisesFisher", log: bool = True):
    pass