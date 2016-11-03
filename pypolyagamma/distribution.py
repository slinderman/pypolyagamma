# Expose some properties of the distribution
import numpy as np
from scipy.special import gamma, gammaln

def _psi_n(x, n, b):
    """
    Compute the n-th term in the infinite sum of
    the Jacobi density.
    """
    return 2**(b-1) / gamma(b) * (-1)**n * \
    np.exp(gammaln(n+b) -
           gammaln(n+1) +
           np.log(2*n+b) -
           0.5 * np.log(2*np.pi*x**3) -
           (2*n+b)**2 / (8.*x))

def _tilt(omega, b, psi):
    """
    Compute the tilt of the PG density for value omega
    and tilt psi.

    :param omega: point at which to evaluate the density
    :param psi: tilt parameter
    """
    return np.cosh(psi/2.0)**b * np.exp(-psi**2/2.0 * omega)


def pgpdf(omega, b, psi, trunc=200):
    """
    Approximate the density log PG(omega | b, psi) using a
    truncation of the density written as an infinite sum.

    :param omega: point at which to evaluate density
    :param b:   first parameter of PG
    :param psi: tilting of PG
    :param trunc: number of terms in sum
    """
    ns = np.arange(trunc)
    psi_ns = np.array([_psi_n(omega, n, b) for n in ns])
    pdf = np.sum(psi_ns, axis=0)

    # Account for tilting
    pdf *= _tilt(omega, b, psi)

    return pdf

def pgmean(b, psi):
    return b / (2.*psi) * np.tanh(psi/2.)

def logistic(x):
    return 1./(1+np.exp(-x))