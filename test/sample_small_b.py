import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
matplotlib.rcParams.update({'font.sans-serif' : 'Helvetica',
                            'axes.labelsize': 9,
                            'xtick.labelsize' : 9,
                            'ytick.labelsize' : 9,
                            'axes.titlesize' : 9})


from hips.plotting.layout import create_figure, create_axis_at_location

from pypolyagamma.pypolyagamma import psi_n
from scipy.special import gamma, gammaln
from scipy.integrate import cumtrapz
plt.ion()


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

def plot_psi(bs=[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], oneminus=False,
             varname="\\eta"):
    """
    In the draft we show that
    the J^*(x | b) density is well approximated by an
    inverse gamma distribution times a scaling factor,
    [1-(2+b)exp{-(4+4b)/2x} + (1+b)(4+b)/2 exp{-(16+8b)/2x} - o(1)]

    We know this factor (if it included its infinitely many terms)
    would have to be a number between 0 and 1. Empirically, it
    seems to decay approximately exponentially. Let's try to come
    up with a parametric approximation for this term.
    """

    xmax = 10.
    xs = np.linspace(0.001, xmax, 1000)

    Nmax = 20
    ns = np.arange(Nmax)

    import palettable
    fig = create_figure(figsize=(3,2), transparent=True)
    ax = create_axis_at_location(fig, 0.6, 0.5, 2., 1.4)
    ax.set_color_cycle(palettable.colorbrewer.sequential.BuGn_9.mpl_colors)
    for b in bs:
        y = 0
        Z = b * gamma(b) * np.exp(-b**2/(2*xs)) * 2**b / gamma(b) / np.sqrt(2*np.pi*xs**3)
        Sns = np.array([psi_n(xs,n,b) for n in ns])
        for trmn in Sns:
            y += trmn / Z

        if oneminus:
            y = 1-y
        ln = ax.plot(xs, y, lw=2, label="b=%.2f" % b)[0]

    ax.set_ylim(-1e-3, 1+1e-3)
    ax.set_xlabel("$%s$" % varname)
    if oneminus:
        ax.set_ylabel("$1-\Psi(%s \\, | \\,  b)$" % varname)
    else:
        ax.set_ylabel("$\Psi(%s \\, | \\,  b)$" % varname)
    plt.legend(loc="upper right", fontsize=8)
    plt.savefig("psi.pdf")

    plt.show()

def fit_psi_cdf(bs=[ 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]):
    xmax = 10.
    xs = np.linspace(0.001, xmax, 1000)

    Nmax = 20
    ns = np.arange(Nmax)

    # gparams = []
    # igparams = []
    params = []
    plt.figure()
    for b in bs:
        print "b=",b
        y = 0
        Z = b * gamma(b) * np.exp(-b**2/(2*xs)) * 2**b / gamma(b) / np.sqrt(2*np.pi*xs**3)
        Sns = np.array([psi_n(xs,n,b) for n in ns])
        for trmn in Sns:
            y += trmn / Z

        ln = plt.plot(xs,(y), label="b=%.2f" % b)[0]
        col = ln.get_color()

        # Make a distribution object for this cdf\
        import scipy.stats as stats
        cdf = 1-y
        # class theoretical_dist(stats.rv_continuous):
        #     def _cdf(self, x, *args):
        #         return np.interp(x, xs, cdf, left=0., right=1.)
        #
        # dist = theoretical_dist()
        # samples = dist.rvs(size=(10000,))

        # Define an objective for fitting a cdf
        def gamma_obj(logprms):
            prms = np.exp(logprms)
            shape, scale = prms[0], prms[1]
            gdist = stats.gamma(shape, scale=scale)
            return gdist.cdf(xs) - cdf

        def invgamma_obj(logprms):
            prms = np.exp(logprms)
            shape, scale = prms[0], prms[1]
            gdist = stats.invgamma(shape, scale=scale)
            return gdist.cdf(xs) - cdf

        def gengamma_obj(logprms):
            prms = np.exp(logprms)
            shape, shape2, scale = prms[0], prms[1], prms[2]
            dist = stats.gengamma(shape, shape2, scale=scale)
            return dist.cdf(xs) - cdf

        def weibull_obj(logprms):
            prms = np.exp(logprms)
            shape, scale = prms[0], prms[1]
            dist = stats.weibull_min(shape, scale=scale)
            return dist.cdf(xs) - cdf

        def exp_obj(logprms):
            prms = np.exp(logprms)
            scale = prms[0]
            dist = stats.expon(scale=scale)
            return dist.cdf(xs) - cdf

        def tanh_obj(prms):
            a,b = prms[0], prms[1]
            f =  (1+ np.tanh(a+b*xs))/2.
            return f - cdf

        # Now fit a few distributions to this and see how well they work
        from scipy.optimize import leastsq
        # gparam = stats.gamma.fit(samples)
        # gparam, _ = leastsq(gamma_obj, np.zeros(2))
        # gparam = np.exp(gparam)
        # gparams.append(gparam)
        # gdist = stats.gamma(gparam[0], scale=gparam[1])
        # plt.plot(xs, 1-gdist.cdf(xs), '--', color=col)

        # igparam = stats.invgamma.fit(samples)
        # igparam, _ = leastsq(invgamma_obj, np.zeros(2))
        # igparam = np.exp(igparam)
        # igparams.append(igparam)
        # igdist = stats.invgamma(igparam[0], scale=igparam[1])
        # plt.plot(xs, 1-igdist.cdf(xs), '-.', color=col)

        # param, _ = leastsq(gengamma_obj, np.zeros(3))
        # param = np.exp(param)
        # params.append(param)
        # dist = stats.gengamma(param[0], param[1], scale=param[2])
        # plt.plot(xs, 1-dist.cdf(xs), '-.', color=col)

        # param, _ = leastsq(weibull_obj, np.zeros(2))
        # param = np.exp(param)
        # params.append(param)
        # dist = stats.weibull_min(param[0], scale=param[1])
        # plt.plot(xs, 1-dist.cdf(xs), '-.', color=col)

        # param, _ = leastsq(exp_obj, np.zeros(1))
        # param = np.exp(param)
        # params.append(param)
        # dist = stats.expon(scale=param[0])
        # plt.plot(xs, 1-dist.cdf(xs), '-.', color=col)

        import ipdb; ipdb.set_trace()
        param, _ = leastsq(tanh_obj, np.zeros(2))
        params.append(param)
        f = (1 + np.tanh(param[0]+param[1]*xs))/2.
        plt.plot(xs, 1-f, '-.', color=col)


    plt.xlabel("x")
    plt.legend(loc="upper right")
    plt.show()

    # Plot the shapes and scales vs b
    # shapes = [gp[0] for gp in gparams]
    # scales = [gp[1] for gp in gparams]
    # plt.figure()
    # plt.subplot(221)
    # plt.plot(bs, shapes)
    # plt.ylabel("shape(b)")
    #
    # plt.subplot(222)
    # plt.plot(bs, scales)
    # plt.ylabel("scale(b)")
    #
    # ishapes = [gp[0] for gp in igparams]
    # iscales = [gp[1] for gp in igparams]
    #
    # plt.subplot(223)
    # plt.plot(bs, ishapes)
    # plt.ylabel("ishape(b)")
    #
    # plt.subplot(224)
    # plt.plot(bs, iscales)
    # plt.ylabel("iscale(b)")

    # ggshapes = [gp[0] for gp in params]
    # ggshapes2 = [gp[1] for gp in params]
    # ggscales = [gp[2] for gp in params]
    # plt.figure()
    # plt.subplot(131)
    # plt.plot(bs, ggshapes)
    # plt.ylabel("GG shape(b)")
    #
    # plt.subplot(132)
    # plt.plot(bs, ggshapes2)
    # plt.ylabel("GG shape2(b)")
    #
    # plt.subplot(133)
    # plt.plot(bs, ggscales)
    # plt.ylabel("GG scale(b)")


def plot_partial_sums(b=0.5):
    xmax = 5.
    xs = np.linspace(1e-3, xmax, 1000)

    Nmax = 5
    ns = np.arange(Nmax)
    Sns = np.array([psi_n(xs,n,b) for n in ns])

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

    plt.figure(figsize=(12,12))
    for i,b in enumerate(bs):
        # Get the terms of the sum
        Sns = np.array([psi_n(xs,n,b) for n in ns])

        # Plot the pdf and the envelopes
        pgpdf = np.sum(Sns, axis=0)
        pgcdf = cumtrapz(pgpdf, xs)
        pglogpdf = np.log(pgpdf)

        # Find the threshold
        # thr = approx_threshold(b)
        # ithr = np.argmin((xs-thr)**2)

        # Get envelopes
        igpdf, iglogpdf, igcdf = invgamma_envelope(xs, b)
        # igausspdf, igausslogpdf, igausscdf = invgauss_envelope(xs[ithr:], b)

        plt.subplot(B,2,2*i+1)
        plt.plot(xs, pgpdf, 'k')
        plt.plot(xs, igpdf, 'b')
        # plt.plot(xs[ithr:], igausspdf,  'r')
        plt.ylabel("p(x | b=%.1f)" % b)
        if i == B-1:
            plt.xlabel("x")

        plt.subplot(B,2,2*i+2)
        plt.plot(xs, pglogpdf, 'k')
        plt.plot(xs, iglogpdf, 'b')
        # plt.plot(xs[ithr:], igausslogpdf, 'r')
        plt.ylim(pglogpdf[-1],5)
        plt.ylabel("log p(x | b=%.1f)" % b)
        if i == B-1:
            plt.xlabel("x")

    # plt.savefig("smallb_pdf_and_envelopes.png")

def plot_acceptance_probability():
    zmin, zmax = -1.5, 1.5
    bmin, bmax = 0, 1
    zs = np.linspace(zmin,zmax,100)
    bs = np.linspace(bmin,bmax,50)
    Zs, Bs = np.meshgrid(zs, bs)
    c = (1+np.exp(-2*abs(Zs)))**Bs
    p = 1./c

    from palettable.colorbrewer.sequential import BuGn_5
    cmap = BuGn_5.mpl_colormap
    fig = create_figure(figsize=(3,1.5), transparent=True)
    ax = create_axis_at_location(fig, 0.6, 0.4, 1.8, .8)
    # ax = fig.add_subplot(111)
    im = ax.imshow(p, cmap=cmap, vmin=0.5, vmax=1., extent=(zmin,zmax,bmax,bmin))
    ax.set_xlabel("$z$")
    ax.set_ylabel("$b$")
    ax.set_yticks((0, 0.25, 0.5, 0.75, 1.0))
    ax.set_title("Acceptance Probability")

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    # divider = make_axes_locatable(plt.gca())
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    cax = create_axis_at_location(fig, 2.5, 0.5, .1, .6)
    cbar = plt.colorbar(im, cax=cax, ticks=(0.5, 0.75, 1.0))

    plt.savefig("acceptance.pdf")
    plt.show()

def quantile_comparison(b):
    """
    Compare our rejection sampling algorithm with the truncated
    sum of gammas algorithm.
    """
    n = 10000
    zs = 1.0 * np.ones(n, dtype=np.float)

    from pypolyagamma import pypolyagamma
    x_rej = pypolyagamma.sample_pg_small_b(b, zs)

    # Sample with the sum of gammas approach
    ppg = pypolyagamma.PyPolyaGamma(np.random.randint(2**16), trunc=200)
    x_sum = np.zeros(n)
    ppg.pgdrawv(b * np.ones(n, dtype=np.float), zs, x_sum)

    from scipy.stats.mstats import mquantiles
    quantiles = np.arange(0.1, 1.0, 0.1)
    q_rej = mquantiles(x_rej, quantiles, alphap=0., betap=1.)
    q_sum = mquantiles(x_sum, quantiles, alphap=0., betap=1.)
    lim = max(q_rej.max(), q_sum.max())

    plt.figure()
    plt.plot(q_rej, q_sum, lw=2)
    plt.plot([0, 1.1*lim], [0, 1.1*lim], '--k')
    plt.show()


# bs = np.linspace(0.05,1.0,9)
# plot_partial_sums(b=0.9)
# plot_envelopes()
# plot_pdf_and_envelope()
# fit_psi_cdf(bs)
plot_psi()
plot_acceptance_probability()
# quantile_comparison(0.5)