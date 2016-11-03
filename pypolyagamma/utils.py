# Expose some properties of the distribution
import numpy as np

from scipy.special import gamma, gammaln, beta
from scipy.linalg.lapack import dpotrs
from scipy.linalg import solve_triangular
from scipy.integrate import simps

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


def sample_gaussian(mu=None,Sigma=None,J=None,h=None):
    # Copied from pybasicbayes
    mean_params = mu is not None and Sigma is not None
    info_params = J is not None and h is not None
    assert mean_params or info_params

    if mu is not None and Sigma is not None:
        return np.random.multivariate_normal(mu,Sigma)
    else:
        L = np.linalg.cholesky(J)
        x = np.random.randn(h.shape[0])
        return solve_triangular(L,x,lower=True,trans='T') \
            + dpotrs(L,h,lower=True)[0]


### Multinomial utils

def get_density(alpha_k, alpha_rest):
    def density(psi):
        return logistic(psi)**alpha_k * logistic(-psi)**alpha_rest \
            / beta(alpha_k,alpha_rest)
    return density

def compute_psi_cmoments(alphas):
    K = alphas.shape[0]
    psi = np.linspace(-10,10,1000)

    mu = np.zeros(K-1)
    sigma = np.zeros(K-1)
    for k in range(K-1):
        density = get_density(alphas[k], alphas[k+1:].sum())
        mu[k] = simps(psi*density(psi),psi)
        sigma[k] = simps(psi**2*density(psi),psi) - mu[k]**2

    return mu, sigma


