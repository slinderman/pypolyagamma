# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

from libc.math cimport exp, pow, sqrt, log, M_PI, erfc, M_SQRT2, M_SQRT1_2
from libc.stdio cimport printf

import numpy as np
cimport numpy as np
from scipy.special import gammaln, gamma

import numpy.random as npr

cdef double __TRUNC = 0.64  # ~2/pi
cdef double __TRUNC_RECIP = 1.0 / __TRUNC # ~pi/2
cdef double

cdef double sample_truncated_invgaussian(double invmu, double t):
    """
    Sample a right-truncated inverse gaussian random variable.
    :param z: inverse mu parameter of the inverse gaussian
    :return:
    """
    cdef double invmu = abs(invmu)

    # x is the sample from the truncated inv gaussian
    cdef double X = t + 1.0

    if (1./t > invmu):
        # mu > t... in this case we can do better apparently?
        cdef double alpha = 0.0
        while r.unif() > alpha:
            # X = t + 1.0;
            # while (X > t)
            # 	X = 1.0 / r.gamma_rate(0.5, 0.5);
            # Slightly faster to use truncated normal.
            cdef  double E1 = r.expon_rate(1.0)
            cdef double E2 = r.expon_rate(1.0)
            while ( E1*E1 > 2 * E2 / t):
                E1 = r.expon_rate(1.0)
                E2 = r.expon_rate(1.0)

            X = 1 + E1 * t
            X = t / (X * X)
            alpha = exp(-0.5 * invmu*invmu * X)

    else:
        cdef double mu = 1.0 / invmu
        # This is equivalent to invgauss_transform(1./Z, 1.)
        # but here we reject as long as X > TRUNC
        while (X > t):

            cdef double z = r.norm(1.0)
            cdef double y = z*z
            cdef double half_mu = 0.5 * mu
            cdef double mu_Y = mu * y

            X = mu + half_mu * mu_Y - half_mu * sqrt(4 * mu_Y + mu_Y * mu_Y)
            if r.unif() > mu / (mu + X):
                X = mu*mu / X

    return X

cpdef invgaussian(double[::1] mus, double[::1] lmbdas):
    """
    Sample inverse gaussian distribution
    Following https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution#Generating_random_variates_from_an_inverse-Gaussian_distribution
    :param mus:
    :param lmbdas:
    :return:
    """
    cdef int n
    cdef int N = mus.shape[0]
    assert lmbdas.shape[0] == N

    # Sample random variables
    cdef double[::1] zs = npr.randn(N)
    cdef double[::1] us = np.random.rand(N)
    cdef double[::1] x = np.zeros(N)

    for n in range(N):
        x[n] = invgaussian_transform(mus[n], lmbdas[n], zs[n], us[n])

    return x

cdef inline double invgaussian_transform(double mu, double lmbda, double z, double u) nogil:
    """
    Sample an inverse Gaussian random variable, IG(mu, lmbda), given
    standard normal random variable, z, and uniform random variable, u.
    :param mu:      mean of IG (>0)
    :param lmbda:   shape of IG (>0)
    :param z:       standard Normal N(0,1) r.v.
    :param u:       uniform r.v. U(0,1).
    :return:        IG(mu, lmbda) distributed r.v.
    """
    cdef double y = z**2
    cdef double ysq = y**2
    cdef double musq = mu**2

    cdef double x = mu
    x += musq * y / 2.0 / lmbda
    x -= mu / 2.0 / lmbda * sqrt(4 * mu * lmbda * y + musq * ysq)

    if u > mu / (mu + x):
        x = musq / x

    return x

cdef inline double gaussian_cdf(x):
    return 0.5 * erfc(-x * M_SQRT1_2)

