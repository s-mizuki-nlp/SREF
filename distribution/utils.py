#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np

def l2_normalize(vec_x: np.ndarray):
    if vec_x.ndim == 1:
        vec_x = vec_x / np.linalg.norm(vec_x)
    elif vec_x.ndim == 2:
        vec_x = vec_x / np.linalg.norm(vec_x, axis=1, keepdims=True)

    return vec_x