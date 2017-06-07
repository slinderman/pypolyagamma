"""
Call the different sample methods
"""
from __future__ import print_function
import sys
import numpy as np
import pypolyagamma as pypolyagamma


# No seed
def test_no_seed(verbose=False):
    ppg = pypolyagamma.PyPolyaGamma()
    v1 = ppg.pgdraw(1., 1.)

    if verbose:
        print(v1)

    return True

# Call the single sample
def test_single_draw(verbose=False):
    np.random.seed(0)
    ppg = pypolyagamma.PyPolyaGamma(np.random.randint(2 ** 16))
    v1 = ppg.pgdraw(1., 1.)

    if verbose:
        print(v1)

    return True

# Sample a vector
def test_vector_draw(verbose=False):
    np.random.seed(0)
    ppg = pypolyagamma.PyPolyaGamma(np.random.randint(2 ** 16))

    # Call the vectorized version
    n = 5
    v2 = np.zeros(n)
    a = 14*np.ones(n, dtype=np.float)
    b = 0*np.ones(n, dtype=np.float)
    ppg.pgdrawv(a, b, v2)

    if verbose:
        print(v2)
    return True

def test_parallel(verbose=False):
    # Call the parallel vectorized version
    np.random.seed(0)

    n = 5
    nthreads = 8
    v3 = np.zeros(n)
    a = 14 * np.ones(n)
    b = 0 * np.ones(n)
    seeds = np.random.randint(2**16, size=nthreads)
    ppgs = [pypolyagamma.PyPolyaGamma(seed) for seed in seeds]
    pypolyagamma.pgdrawvpar(ppgs, a, b, v3)

    if verbose:
        print(v3)
    return True

def ks_test(b=1.0, c=0.0, N_smpls=10000, N_pts=10000):
    """
    Kolmogorov-Smirnov test. We can't calculate the CDF exactly,
    but we can do a pretty good job with numerical integration.
    """
    # Estimate the true CDF
    oms = np.linspace(1e-5, 3.0, N_pts)
    pdf = pypolyagamma.pgpdf(oms, b, c, trunc=200)
    cdf = lambda x: min(np.trapz(pdf[oms < x], oms[oms < x]), 1.0)

    # Draw samples
    ppg = pypolyagamma.PyPolyaGamma(np.random.randint(2 ** 16))
    smpls = 1e-3 * np.ones(N_smpls)
    ppg.pgdrawv(b * np.ones(N_smpls),  c * np.ones(N_smpls), smpls)

    # TODO: Not sure why this always gives a p-value of zero
    from scipy.stats import kstest
    print(kstest(smpls, cdf))

# test samples against the density
def test_density(b=1.0, c=0.0, N_smpls=10000, plot=False):
    # Draw samples from the PG(1,0) distributions
    ppg = pypolyagamma.PyPolyaGamma(np.random.randint(2 ** 16))
    smpls = np.zeros(N_smpls)
    ppg.pgdrawv(np.ones(N_smpls), np.zeros(N_smpls), smpls)

    # Compute the empirical PDF
    bins = np.linspace(0, 2.0, 50)
    centers = 0.5 * (bins[1:] + bins[:-1])
    p_centers = pypolyagamma.pgpdf(centers, b, c)
    empirical_pdf, _ = np.histogram(smpls, bins, normed=True)

    # Check that the empirical pdf is close to the true pdf
    err = (empirical_pdf - p_centers) / p_centers
    assert np.all(np.abs(err) < 10.0), \
        "Max error of {} exceeds tolerance of 5.0".format(abs(err).max())

    if plot:
        import matplotlib.pyplot as plt
        plt.hist(smpls, bins=50, normed=True, alpha=0.5)

        # Plot high resolution density
        oms = np.linspace(1e-3, 2.0, 1000)
        pdf = pypolyagamma.pgpdf(oms, b, c)
        plt.plot(oms, pdf, '-b', lw=2)
        plt.show()
    return True


if __name__ == "__main__":
    verbose = len(sys.argv) > 1 and (sys.argv[1] == '-v' or sys.argv[1] == "--verbose")
    assert test_no_seed(verbose)
    assert test_single_draw(verbose)
    assert test_vector_draw(verbose)
    assert test_parallel(verbose)
    assert test_density()
    # ks_test()
    print("Tests passed!")
