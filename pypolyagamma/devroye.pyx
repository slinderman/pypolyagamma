# distutils: extra_link_args = -fopenmp
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

from libc.math cimport exp, pow, sqrt, log, M_PI
from libc.stdio cimport printf

from cython.parallel import prange, parallel
from libcpp.vector cimport vector

from openmp cimport omp_get_num_threads, omp_get_thread_num, omp_get_max_threads

import numpy as np
cimport numpy as np
from scipy.special import gammaln, gamma

import numpy.random as npr

# The truncation is set such that the inverse gaussian
# provides an upper bound for the left tail and the
# exponential provides an upper bound on the right.
cdef double __TRUNC = 0.64


cpdef double draw_like_devroye(int n, double z):
    cdef int i = 0
    cdef double sum = 0.0
    for i in range(n):
        sum += _std_draw_like_devroye(z)


cpdef double _std_draw_like_devroye(double z):
    # Change the parameter.
    z = abs(z) * 0.5

    # Now sample 0.25 * J^*(1, Z := Z/2).
    cdef double rate = 0.125 * M_PI**2 + 0.5 * z * z
    # ... Problems with large Z?  Try using q_over_p.
    # double p  = 0.5 * __PI * exp(-1.0 * fz * __TRUNC) / fz;
    # double q  = 2 * exp(-1.0 * Z) * pigauss(__TRUNC, Z);

    cdef double X = 0.0
    cdef double S = 1.0
    cdef double u = 0.0
    cdef int niter = 0
    cdef bool go = True

    while True:
    #if (r.unif() < p/(p+q))
        if r.unif() < _mixture_threshold(z):
            X = __TRUNC + r.expon_rate(rate)
        else:
            X = sample_truncated_invgaussian(z, __TRUNC)

        # Build up the series for the pdf (one term at a time)
        S = _a(0, X)
        u = r.unif() * S
        niter = 0
        reject = False

        # Check whether the uniform r.v. is less than the alternating bound
        while not reject:
            niter += 1
            if niter % 2 == 1:
                # Odd => lower bound
                S = S - _a(niter, X)
                if u <= S:
                    return 0.25 * X

            else:
                # Even implies upper bound
                S = S + _a(niter, X)
                if u > S:
                    reject = True

    # Need Y <= S in event that Y = S, e.g. when X = 0.

# Helper functions
cdef double _a(int n, double x):
    cdef double K = (n + 0.5) * M_PI
    cdef double y = 0
    cdef double expnt

    if x > __TRUNC:
        y = K * exp( -0.5 * K*K * x )
    elif x > 0:
        expnt = -1.5 * (log(0.5 * M_PI)  + log(x)) + log(K) - 2.0 * (n+0.5)*(n+0.5) / x
        y = exp(expnt)
        #y = pow(0.5 * __PI * x, -1.5) * K * exp( -2.0 * (n+0.5)*(n+0.5) / x);
        #^- unstable for small x?

    return y


cdef double _mixture_threshold(double z):
    cdef double t = __TRUNC

    cdef double fz = 0.125 * M_PI*M_PI + 0.5 * z * z;
    cdef double b = sqrt(1.0 / t) * (t * z - 1)
    cdef double a = sqrt(1.0 / t) * (t * z + 1) * -1.0

    cdef double x0 = log(fz) + fz * t;
    cdef double xb = x0 - z + gaussian_cdf(b)
    cdef double xa = x0 + z + gaussian_cdf(a)
    cdef double qdivp = 4.0 / M_PI * ( exp(xb) + exp(xa) )

    return 1.0 / (1.0 + qdivp)
