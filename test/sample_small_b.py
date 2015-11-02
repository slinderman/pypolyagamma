import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import pypolyagamma as ppg
from scipy.special import gamma, gammaln
from scipy.integrate import cumtrapz
plt.ion()

# Use the identity Gam(b,c) ~ Gam(b+1, c) * U^{1/b}
# where ~ means equal in distribution.

# Compute the moments of the distribution over Y=U^{1/b}
# E[y] = b/(b+1)
# E[y^2] = b/(b+2)
# Var[y] = b / (b^3 + 4b^2 + 5b + 2)

def mean_y(b):
    return b/(b+1.)

def var_y(b):
    return b / (b**3 + 4.*b**2 + 5.*b + 2.)

def std_y(b):
    return np.sqrt(var_y(b))

# Def compute the terms of PG gamma rv's
def ds(ns):
    return np.pi**2 / 2.0 * (ns + 0.5)**2

def dssq(ns):
    return ds(ns)**2

def var_S(b):
    return var_y(b) * (b**2 + 3*b + 2.) * np.sum(1./dssq(np.arange(20)))

def std_S(b):
    return np.sqrt(var_S(b))

# Look at terms in the alternating sum
def Sn(x,n,b):
    return 2**b / gamma(b) * (-1)**n * \
    np.exp(gammaln(n+b) -
           gammaln(n+1) +
           np.log(2*n+b) -
           0.5 * np.log(2*np.pi*x**3) -
           (2*n+b)**2 / (2.*x))


# plt.figure()
# bs = np.linspace(1e-3, 1.0)
# plt.errorbar(bs, mean_y(bs), std_y(bs))

# plt.figure()
# ns = np.arange(20)
# plt.plot(ns, 1./dssq(ns))
#
# plt.figure()
# bs = np.linspace(1e-3, 1.0)
# plt.plot(bs, std_S(bs))

def compare_samples():
    nthreads = 8
    seeds = np.random.randint(2**16, size=nthreads)
    ppgs = [ppg.PyPolyaGamma(seed) for seed in seeds]

    # Sample PG(0.5, 0)
    M = 10000
    b0 = 0.5
    om1 = np.zeros(M)
    ppg.pgdrawvpar(ppgs, b0 * np.ones(M), np.zeros(M), om1)

    # Sample PG(1.5, 0)
    om2 = np.zeros(M)
    ppg.pgdrawvpar(ppgs, (b0 + 1) * np.ones(M), np.zeros(M), om2)

    # Sample G(0.5, 2*pi^2 * (-1/2)**2
    # g0 = np.random.gamma(b0, 1./( 2*np.pi**2 * (0+0.5)**2), size=(M,))
    # g1 = np.random.gamma(b0, 1./( 2*np.pi**2 * (1+0.5)**2), size=(M,))
    # g2 = np.random.gamma(b0, 1./( 2*np.pi**2 * (2+0.5)**2), size=(M,))
    # g3 = np.random.gamma(b0, 1./( 2*np.pi**2 * (3+0.5)**2), size=(M,))
    # g4 = np.random.gamma(b0, 1./( 2*np.pi**2 * (4+0.5)**2), size=(M,))
    # g5 = np.random.gamma(b0, 1./( 2*np.pi**2 * (5+0.5)**2), size=(M,))
    gs = [np.random.gamma(b0, 1./( 2*np.pi**2 * (n+0.5)**2), size=(M,)) for n in xrange(10)]

    plt.figure()
    plt.hist(om1, 50, normed=True, color='b', alpha=0.5)
    # plt.hist((b0 / (1+b0)) * om2, 50, normed=True, color='r', alpha=0.5)
    for n in xrange(5):
        plt.hist(np.sum(gs[:n+1], axis=0), 50, normed=True, alpha=0.5)


# def plot_terms():
xmax = 5.
xs = np.linspace(1e-3, xmax, 1000)
b = 0.5
Nmax = 20
ns = np.arange(Nmax)
Sns = np.array([Sn(xs,n,b) for n in ns])

# # Plot the partial sums
# plt.figure()
# cm = get_cmap("Greys")
# for n in xrange(1,Nmax):
#     plt.plot(xs, np.sum(Sns[:n], axis=0),
#              color=cm(np.clip(float(n)/Nmax, 0.3, 0.7)),
#              label="n=%d" % n)
# plt.legend()

# Plot the log pdf
# It should not be concave
pgpdf = np.sum(Sns, axis=0)
pgcdf = cumtrapz(pgpdf, xs)
pglogpdf = np.log(pgpdf)

# Compute the density of an inverse gamma kernel
alpha = 0.5
beta = b**2 / 2.
iglogpdf = b*np.log(2) + alpha * np.log(beta) - gammaln(alpha) + -(alpha+1) * np.log(xs) - beta/xs
igpdf = np.exp(iglogpdf)
igcdf = cumtrapz(igpdf / 2.0**b, xs)

# Compute a gamma distribution envelope
alpha = b
beta = np.pi**2 / 8.
glogpdf = b * np.log(6./np.pi) + alpha * np.log(beta) - gammaln(alpha) + (alpha-1) * np.log(xs) - beta*xs
gpdf = np.exp(glogpdf)
gcdf = cumtrapz(gpdf / (6/np.pi)**b, xs)

plt.figure()
plt.subplot(131)
plt.plot(xs, pgpdf, 'b')
plt.plot(xs, igpdf, 'r')
# plt.plot(xs, gpdf,  'g')
plt.subplot(132)
plt.plot(xs, pglogpdf, 'b')
plt.plot(xs, iglogpdf, 'r')
plt.plot(xs, glogpdf, 'g')
plt.subplot(133)
plt.plot(xs[1:], pgcdf, 'b')
plt.plot(xs[1:], igcdf, 'r')
plt.plot(xs[1:], gcdf, 'g')

# Plot the terms in the sum
# Check if they are decreasing
def _check_decreasing(X):
    # Make sure each row is smaller than the one before
    N,T = X.shape
    dec = np.ones(T, dtype=np.bool)
    for n in xrange(1,N):
        dec &= (X[n] <= X[n-1])
    return dec

isdec = _check_decreasing(np.abs(Sns))
print "Decreasing until cdf = ", pgcdf[np.amin(np.where(~isdec)[0])]

# plt.figure()
# for n in xrange(Nmax):
#     plt.semilogy(np.abs(Sns[n]))
