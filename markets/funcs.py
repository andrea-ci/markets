# -*- coding: utf-8 -*-
import warnings
import math
import numpy as np
from numba import njit

warnings.filterwarnings('error')

@njit
def f_eval(t, tc, m, w, a, b, c1, c2):
    """Evaluates LPPLS function for a given time point."""

    fp = a + math.pow(tc - t, m) * (b + ((c1 * math.cos(w * math.log(tc - t))) + \
        (c2 * math.sin(w * math.log(tc - t)))))

    return fp

@njit
def get_c(c1, c2):
    """Evaluates C parameter given C1 and C2."""

    c = c1 / math.cos(math.atan(c2 / c1))

    return c

@njit
def solve_matrix_equation(obs, tc, m, w):
    """Optimization problem for linear parameters: a, b, c1, c2.

    For details see:
        Filimonov & Sornette, A stable and robust calibration scheme of the
        log-periodic power law model, 2013.
    """

    n_points = obs.shape[1]

    time = obs[0]
    price = obs[1]

    # @TODO make taking tc - t or |tc - t| configurable
    dT = np.abs(tc - time)
    phase = np.log(dT)

    fi = np.power(dT, m)
    gi = fi * np.cos(w * phase)
    hi = fi * np.sin(w * phase)

    fi_pow_2 = np.power(fi, 2)
    gi_pow_2 = np.power(gi, 2)
    hi_pow_2 = np.power(hi, 2)

    figi = np.multiply(fi, gi)
    fihi = np.multiply(fi, hi)
    gihi = np.multiply(gi, hi)

    yi = price
    yifi = np.multiply(yi, fi)
    yigi = np.multiply(yi, gi)
    yihi = np.multiply(yi, hi)

    matrix_1 = np.array([
        [n_points,      np.sum(fi),       np.sum(gi),       np.sum(hi)],
        [np.sum(fi),    np.sum(fi_pow_2), np.sum(figi),     np.sum(fihi)],
        [np.sum(gi),    np.sum(figi),     np.sum(gi_pow_2), np.sum(gihi)],
        [np.sum(hi),    np.sum(fihi),     np.sum(gihi),     np.sum(hi_pow_2)]
    ])

    matrix_2 = np.array([
        [np.sum(yi)],
        [np.sum(yifi)],
        [np.sum(yigi)],
        [np.sum(yihi)]
    ])

    # Compute the linear solution.
    sol = np.linalg.solve(matrix_1, matrix_2)
    a, b, c1, c2 = sol[:, 0]

    return a, b, c1, c2

@njit
def get_oscillations(w, tc, t1, t2):
    return w / (2.0 * math.pi) * math.log((tc - t1) / (tc - t2))

@njit
def get_damping(m, w, b, c):
    return m * abs(b) / (w * abs(c))
