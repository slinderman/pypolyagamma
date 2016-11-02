
from libc.math cimport exp, pow, sqrt
from libc.stdio cimport printf

from cython.parallel import prange, parallel

import numpy as np
cimport numpy as np
from scipy.special import gammaln, gamma

import numpy.random as npr

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

def _one_minus_psi_coeffs(b, K):
    """
    Compute the coefficients in the 1-\Psi(x|b) coefficients
    """
    cs = np.zeros(K)
    cs[0] = 1.0
    for k in range(1,K):
        js = np.arange(1,k)
        jprod = np.prod(js+b) if len(js) > 0 else 1.0
        kfact = np.prod(np.arange(1,k+1))
        cs[k] = (-1)**k * jprod / kfact * (2*k+b)
    return cs


cdef inline double _one_minus_psi(double x, double b) nogil:
    """
    Compute 1 - \Psi(x | b)
    :param x:
    :param b:
    """
    cdef double omp = 1.0
    omp -= (2.0+b) * exp(-2.*(b+1.)/x)
    omp += (1.0+b)*(4.0+b)/2.0 * exp(-4.*(b+2.)/x)
    omp -= (2.0+b)*(1.0+b)*(6.0+b)/6.0 * exp(-6.*(b+3.)/x)
    omp += (3.0+b)*(2.0+b)*(1.0+b)*(8.0+b)/24.0 * exp(-8.*(b+4.)/x)
    omp -= (4.0+b)*(3.0+b)*(2.0+b)*(1.0+b)*(10.0+b)/120.0 * exp(-10.*(b+5.)/x)
    return omp

cdef inline double _one_minus_psi_given_coeffs(double x, double b, double[::1] cs, int K) nogil:
    """
    Compute 1 - \Psi(x | b)
    :param x:
    :param b:
    """
    cdef double omp = 1.0
    cdef int n
    for n in range(1,K):
        omp += cs[n] * exp(-2.*n*(b+n)/x)
    return omp


cpdef one_minus_psi(double[::1] xs, double b, int K=6):
    """
    Compute 1 - \Psi(x | b)
    :param x:
    :param b:
    """
    cdef int N = xs.shape[0]
    cdef double[::1] omp = np.zeros(N)
    cdef int n
    cdef double[::1] cs = _one_minus_psi_coeffs(b, K)
    for n in prange(N, nogil=True):
        # omp[n] = _one_minus_psi(xs[n], b)
        omp[n] = _one_minus_psi_given_coeffs(xs[n], b, cs, K)

    return np.array(omp)

cpdef sample_pg_small_b(double b, double[::1] zs, int K=6, int maxiter=10):
    """
    Sample a vector of PG(b,z) random variables for b < 1.
    Rejection sample from an inverse Gaussian proposal distribution.
    :param b: shape parameter
    :param zs: tilting parameter
    :param K: truncation of 1-\Psi(x|b) calculation
    :param maxiter: max number of rejection sampling iterations
    :return: vector of PG(b,z) rv's the same shape as zs.
    """
    js = sample_jacobi_small_b(b, np.divide(zs, 2.0), K=K, maxiter=maxiter)
    return js / 4.0

cpdef sample_jacobi_small_b(double b, double[::1] zs, int K=6, int maxiter=10):
    """
    Sample a vector of J^*(b,z) random variables for b < 1.
    Rejection sample from an inverse Gaussian proposal distribution.
    :param b: shape parameter
    :param zs: tilting parameter
    :param K: truncation of 1-\Psi(x|b) calculation
    :param maxiter: max number of rejection sampling iterations
    :return: vector of J^*(b,z) rv's the same shape as zs.
    """
    # First sample a buffer of inverse Gaussian r.v.'s
    # We can amortize the cost by sampling these all at once.
    # Since we're rejection sampling, we don't know how many
    # we will need, a priori. The expected number of required
    # samples is roughly 2^b
    cdef int N = zs.shape[0]
    cdef int B = (int) (np.power(2.0, b) * N)

    # Sample a buffer of standard normal and uniform r.v.'s
    cdef double[::1] ns = np.random.randn(B)
    cdef double[::1] us = np.random.rand(B)
    cdef double[::1] ts = np.random.rand(B)

    # Initialize output
    cdef double[::1] xs = np.zeros(N)

    # Special case b==0
    if b == 0:
        return np.zeros(N)

    # Compute the mean and shape of the IG proposal
    cdef double[::1] mu = np.divide(b, np.abs(zs))
    cdef double[::1] lmbda = b**2 * np.ones(B)

    # Precompute coefficients for computing 1-\Psi(x|b)
    cdef double[::1] cs = _one_minus_psi_coeffs(b, K)

    # Rejection sample
    cdef int n = 0
    cdef int i = 0
    cdef double ig = 0
    cdef int success = False
    cdef int niter = 0
    for n in range(N):
        niter = 0
        success = False

        while (not success) and (niter < maxiter):
            # Sample IG(mu, lmbda) proposal
            ig = invgaussian_transform(mu[n], lmbda[n], ns[i], us[i])

            # Check for success
            success = ts[i] < _one_minus_psi_given_coeffs(ig, b, cs, K)
            if success:
                xs[n] = ig

            # Replenish buffer if necessary
            i += 1
            if i >= B:
                ns = np.random.randn(B)
                us = np.random.rand(B)
                ts = np.random.rand(B)
                i = 0

            # Update iteration count
            niter += 1

        # printf("n: %d \t niter: %d \t ig: %.3f\n", n, niter, ig)

    return np.array(xs)


def psi_n(x,n,b):
    return 2**(b-1) / gamma(b) * (-1)**n * \
    np.exp(gammaln(n+b) -
           gammaln(n+1) +
           np.log(2*n+b) -
           0.5 * np.log(2*np.pi*x**3) -
           (2*n+b)**2 / (8.*x))

def pgpdf(omega, b, psi, trunc=20):
    """
    Approximate the density log PG(omega | b, psi) using a
    truncation of the density written as an infinite sum.

    :param omega: point at which to evaluate density
    :param b:   first parameter of PG
    :param psi: tilting of PG
    :param trunc: number of terms in sum
    """
    # TODO: Include tilt
    ns = np.arange(trunc)
    psi_ns = np.array([psi_n(omega,n,b) for n in ns])
    pdf = np.sum(psi_ns, axis=0)
    return pdf

def pgmean(b, psi):
    return b / (2.*psi) * np.tanh(psi/2.)