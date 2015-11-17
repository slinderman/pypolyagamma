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

def invgamma_envelope(xs, b):
    # Compute the density of an inverse gamma kernel
    alpha = 0.5
    beta = b**2 / 2.
    iglogpdf = b*np.log(2) + alpha * np.log(beta) - gammaln(alpha) + -(alpha+1) * np.log(xs) - beta/xs
    igpdf = np.exp(iglogpdf)
    igcdf = cumtrapz(igpdf / 2.0**b, xs) if np.size(xs) > 1 else None
    return igpdf, iglogpdf, igcdf

def gamma_envelope(xs, b):
    # Compute a gamma distribution envelope
    alpha = b
    beta = np.pi**2 / 8.
    glogpdf = b * np.log(6./np.pi) + alpha * np.log(beta) - gammaln(alpha) + (alpha-1) * np.log(xs) - beta*xs
    gpdf = np.exp(glogpdf)
    gcdf = cumtrapz(gpdf / (6/np.pi)**b, xs) if np.size(xs) > 1 else None
    return gpdf, glogpdf, gcdf

# Compute an inverse Gaussian envelope
def invgauss_envelope(xs, b):
    z = 1.
    mu = b / z
    lam = b**2
    igausslogpdf = b * np.log(2.) + 0.5 * np.log(lam) -0.5*np.log(2*np.pi*xs**3) - lam * (xs-mu)**2 / (2*mu**2 * xs)
    igausspdf = np.exp(igausslogpdf)
    igausscdf = cumtrapz(igausspdf / (2**b), xs) if np.size(xs) > 1 else None
    return igausspdf, igausslogpdf, igausscdf

def plot_terms_inside_brackets(bs=[0.5]):
    """
    This is a horrible name, but in the draft we show that
    the J^*(x | b) density is well approximated by an
    inverse gamma distribution times a scaling factor,
    [1-(2+b)exp{-(4+4b)/2x} + (1+b)(4+b)/2 exp{-(16+8b)/2x} - o(1)]

    We know this factor (if it included its infinitely many terms)
    would have to be a number between 0 and 1. Empirically, it
    seems to decay approximately exponentially. Let's try to come
    up with a parametric approximation for this term.
    """
    xmax = 5.
    xs = np.linspace(1e-3, xmax, 1000)

    plt.figure()
    for b in bs:
        trm0 = 1
        trm1 = -(2+b) * np.exp(-(4+4*b)/(2*xs))
        trm2 = (1+b)*(4+b)/2 * np.exp(-(16+8*b)/(2*xs))

        y = trm0 + trm1 + trm2
        plt.plot(xs,np.log(y), color='b')

        # plt.plot(xs, np.log(y) + -b**2 / xs, '-r', lw=2)
        plt.plot(xs,  -b**2 / xs, '-r', lw=2)
    plt.plot(xs,  -1.5 * np.log(xs), '-g', lw=2)
    plt.plot(xs, np.log(y) + -b**2 / xs + -1.5 * np.log(xs), '-k', lw=2)

    plt.ylim(-8,4)
    # plt.legend(loc="upper right")
    plt.show()

def plot_partial_sums(b=0.5):
    xmax = 5.
    xs = np.linspace(1e-3, xmax, 1000)

    Nmax = 5
    ns = np.arange(Nmax)
    Sns = np.array([Sn(xs,n,b) for n in ns])

    # Check if partial sums are decreasing
    pgpdf = np.sum(Sns, axis=0)
    pgcdf = cumtrapz(pgpdf, xs)
    def _check_decreasing(X):
        # Make sure each row is smaller than the one before
        N,T = X.shape
        dec = np.ones(T, dtype=np.bool)
        for n in xrange(1,N):
            dec &= (X[n] <= X[n-1])
        return dec

    isdec = _check_decreasing(np.abs(Sns))
    print "Decreasing until cdf = ", pgcdf[np.amin(np.where(~isdec)[0])]


    # Plot the partial sums
    plt.figure()
    cm = get_cmap("Greys")
    for n in xrange(1,Nmax):
        plt.plot(xs, np.sum(Sns[:n], axis=0),
                 color=cm(np.clip(float(n)/Nmax, 0.3, 0.7)),
                 label="n=%d" % n)
    plt.legend()

    plt.figure()
    cm = get_cmap("Greys")
    for n in xrange(Nmax):
        plt.plot(xs, Sns[n],
                 color=cm(np.clip(1-float(n)/Nmax, 0.3, 0.7)),
                 label="n=%d" % n)
    plt.legend()

def approx_threshold(b):
    # b=0.10, thr=0.2
    # b=1.00, thr=2.0
    # thr = mb + c
    # m = (2-0.2) / (1-0.1)
    m = 2.0
    # c = thr - m*b = 2.0 - m * 1.0 = 0.0
    c = 0.0
    return m*b + c

def plot_envelopes():

    xmax = 5.
    xs = np.linspace(1e-3, xmax, 1000)
    bs = np.linspace(1e-1, 1.00, 10)
    ints = []

    def _find_intersect(y1,y2,start=0,stop=-1):
        return start + np.argmin((y1[start:stop]-y2[start:stop])**2)

    plt.figure()
    for i,b in enumerate(bs):
        # Get envelopes
        igpdf, iglogpdf, igcdf = invgamma_envelope(xs, b)
        igausspdf, igausslogpdf, igausscdf = invgauss_envelope(xs, b)
        int = _find_intersect(igpdf, igausspdf,
                              start=np.argmin((xs-0.05)**2),
                              stop=np.argmin((xs-2.0)**2))
        ints.append(int)
        # alpha = 0.1 + (0.9 * i)/len(bs)
        alpha = 0.5
        plt.plot(xs, igpdf, '-b', alpha=alpha)
        plt.plot(xs, igausspdf, '-r', alpha=alpha)
        plt.plot(xs[int], igpdf[int], 'ko')

    # Plot the approximate thresholds
    bbs = np.linspace(bs[0], bs[-1])
    thr = approx_threshold(bbs)
    ythr = np.array([invgauss_envelope(t,b) for t,b in zip(thr,bbs)])
    plt.plot(thr, ythr, '-k' )

    plt.ylim(0,1)
    plt.ylabel("l(x | b) and r(x|b)")
    plt.xlabel("x")
    plt.savefig("smallb_envelopes.png")
    plt.show()

    plt.figure()
    plt.plot(bs, xs[ints], 'ok')
    plt.plot(bbs, thr, '-k')
    plt.show()

    plt.savefig("smallb_crossover.png")

def plot_pdf_and_envelope():
    xmax = 5.
    xs = np.linspace(1e-3, xmax, 1000)
    Nmax = 20
    ns = np.arange(Nmax)
    B = 5
    bs = np.linspace(0.1, 0.95, B)

    slopes = approx_log_pgpdf_slope(bs)

    plt.figure(figsize=(12,12))
    for i,b in enumerate(bs):
        # Get the terms of the sum
        Sns = np.array([Sn(xs,n,b) for n in ns])

        # Plot the pdf and the envelopes
        pgpdf = np.sum(Sns, axis=0)
        pgcdf = cumtrapz(pgpdf, xs)
        pglogpdf = np.log(pgpdf)

        # Find the threshold
        thr = approx_threshold(b)
        ithr = np.argmin((xs-thr)**2)

        # Get envelopes
        igpdf, iglogpdf, igcdf = invgamma_envelope(xs[:ithr], b)
        igausspdf, igausslogpdf, igausscdf = invgauss_envelope(xs[ithr:], b)

        plt.subplot(B,2,2*i+1)
        plt.plot(xs, pgpdf, 'k')
        plt.plot(xs[:ithr], igpdf, 'b')
        plt.plot(xs[ithr:], igausspdf,  'r')
        plt.ylabel("p(x | b=%.1f)" % b)
        if i == B-1:
            plt.xlabel("x")

        plt.subplot(B,2,2*i+2)
        plt.plot(xs, pglogpdf, 'k')
        plt.plot(xs[:ithr], iglogpdf, 'b')
        plt.plot(xs[ithr:], igausslogpdf, 'r')
        plt.ylim(pglogpdf[-1],5)
        plt.ylabel("log p(x | b=%.1f)" % b)
        if i == B-1:
            plt.xlabel("x")

    # plt.savefig("smallb_pdf_and_envelopes.png")


    # plt.subplot(133)

    # plt.plot(xs[1:], pgcdf, 'k')
    # plt.plot(xs[:ithr-1], igcdf, 'b')
    # plt.plot(xs[ithr:-1], igcdf[ithr] + igausscdf, 'r')

def plot_pdf_and_exp_envelope():
    xmax = 5.
    xs = np.linspace(1e-3, xmax, 1000)
    Nmax = 20
    ns = np.arange(Nmax)
    B = 5
    bs = np.linspace(0.1, 0.95, B)

    slopes, offsets = approx_log_pgpdf_slope(bs)

    plt.figure(figsize=(12,12))
    for i,b in enumerate(bs):
        # Get the terms of the sum
        Sns = np.array([Sn(xs,n,b) for n in ns])

        # Plot the pdf and the envelopes
        pgpdf = np.sum(Sns, axis=0)
        pgcdf = cumtrapz(pgpdf, xs)
        pglogpdf = np.log(pgpdf)

        # Find the point of inflection of the log pdf
        thr = b**2/6.
        ithr = np.argmin((xs-thr)**2)

        # Find the gradient o the log pdf
        dlogpdf = -1.5 * xs**(-1) + b**2/2 * xs**(-2.)
        d2logpdf = 1.5 * xs**(-2) - b**2 * xs**(-3.)

        # Get envelopes
        igpdf, iglogpdf, igcdf = invgamma_envelope(xs, b)

        # Exponential envelope for right tail
        logexppdf = slopes[i] * xs + offsets[i]
        exppdf = np.exp(logexppdf)

        plt.subplot(B,2,2*i+1)
        plt.plot(xs, pgpdf, 'k')
        plt.plot(xs, igpdf, 'b')
        plt.plot(xs, exppdf,  'g')
        plt.ylabel("p(x | b=%.1f)" % b)
        if i == B-1:
            plt.xlabel("x")

        plt.subplot(B,2,2*i+2)
        plt.plot(xs, pglogpdf, 'k')
        plt.plot(xs, iglogpdf, 'b')
        plt.plot(xs, logexppdf, 'g')
        # plt.plot(xs, dlogpdf, color='orange')
        # plt.plot(xs, d2logpdf, color='purple')
        plt.plot(xs, np.zeros_like(xs), '--k', alpha=0.5)
        # plt.ylim(pglogpdf[-1],5)
        plt.ylim(-9,5)
        plt.ylabel("log p(x | b=%.1f)" % b)
        if i == B-1:
            plt.xlabel("x")

    # plt.savefig("smallb_pdf_and_envelopes.png")


    # plt.subplot(133)

    # plt.plot(xs[1:], pgcdf, 'k')
    # plt.plot(xs[:ithr-1], igcdf, 'b')
    # plt.plot(xs[ithr:-1], igcdf[ithr] + igausscdf, 'r')

def approx_log_pgpdf_slope(bs):
    xs = np.array([0.8, 5.])
    Nmax = 20
    ns = np.arange(Nmax)

    offsets = np.zeros_like(bs)
    slopes = np.zeros_like(bs)
    for i,b in enumerate(bs):
        # Get the terms of the sum
        Sns = np.array([Sn(xs,n,b) for n in ns])
        pgpdf = np.sum(Sns, axis=0)
        pglogpdf = np.log(pgpdf)

        slopes[i] = (pglogpdf[1] - pglogpdf[0]) / (xs[1] - xs[0])
        offsets[i] = pglogpdf[0] - slopes[i]*xs[0]

    print "Slopes of log pdf: "
    for b,slope,offset in zip(bs, slopes, offsets):
        print "b: ", b, "\t slope: ", slope, "\t off: ", offset


    return slopes, offsets

plot_terms_inside_brackets()
# plot_partial_sums(b=0.9)
# plot_envelopes()
# plot_pdf_and_envelope()
# plot_pdf_and_exp_envelope()
# approx_log_pgpdf_slope(np.linspace(0.1, 0.95, 5))