import abc
import numpy as np
import numpy.random as npr
from scipy.special import gammaln
from scipy.sparse import csr_matrix
from scipy.misc import logsumexp

# Name collision in pypolyagamma.pypolyagamma
# Use relative paths instead (ugh).
from .utils import logistic, sample_gaussian, psi_to_pi, compute_psi_cmoments
from . import get_omp_num_threads, pgdrawvpar, PyPolyaGamma

class _PGLogisticRegressionBase(object):
    """
    A base class for the emission matrix, C.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, D_out, D_in, A=None,
                 mu_A=0., sigmasq_A=1.,
                 b=None, mu_b=0., sigmasq_b=10.):
        """
        :param D_out: Observation dimension
        :param D_in: Latent dimension
        :param A: Initial NxD emission matrix
        :param sigmasq_A: prior variance on C
        :param b: Initial Nx1 emission matrix
        :param sigmasq_b: prior variance on b
        """
        self.D_out, self.D_in, \
        self.mu_A, self.sigmasq_A, \
        self.mu_b, self.sigmasq_b = \
            D_out, D_in, mu_A, sigmasq_A, mu_b, sigmasq_b


        if np.isscalar(mu_b):
            self.mu_b = mu_b * np.ones(D_out)

        if np.isscalar(sigmasq_b):
            self.sigmasq_b = sigmasq_b * np.ones(D_out)

        if np.isscalar(mu_A):
            self.mu_A = mu_A * np.ones((D_out, D_in))
        else:
            assert mu_A.shape == (D_out, D_in)

        if np.isscalar(sigmasq_A):
            self.sigmasq_A = np.array([sigmasq_A * np.eye(D_in) for _ in range(D_out)])
        else:
            assert sigmasq_A.shape == (D_out, D_in, D_in)

        if A is not None:
            assert A.shape == (self.D_out, self.D_in)
            self.A = A
        else:
            self.A = np.zeros((self.D_out, self.D_in))
            for d in range(self.D_out):
                self.A[d] = npr.multivariate_normal(self.mu_A[d], self.sigmasq_A[d])

        if b is not None:
            assert b.shape == (self.D_out, 1)
            self.b = b
        else:
            # self.b = np.sqrt(sigmasq_b) * npr.rand(self.D_out, 1)
            self.b = self.mu_b[:,None] + np.sqrt(self.sigmasq_b[:,None]) * npr.randn(self.D_out,1)

        # Initialize Polya-gamma samplers
        num_threads = get_omp_num_threads()
        seeds = npr.randint(2 ** 16, size=num_threads)
        self.ppgs = [PyPolyaGamma(seed) for seed in seeds]

    @abc.abstractmethod
    def a_func(self, data):
        raise NotImplementedError

    @abc.abstractmethod
    def b_func(self, data):
        raise NotImplementedError

    @abc.abstractmethod
    def log_c_func(self, data):
        raise NotImplementedError

    def _elementwise_log_likelihood(self, xy, mask=None):
        if isinstance(xy, tuple):
            x,y = xy
        elif isinstance(xy, np.ndarray):
            x,y = xy[:,:self.D_in], xy[:,self.D_in:]
        else:
            raise NotImplementedError

        psi = x.dot(self.A.T) + self.b.T
        ll = self.log_c_func(y) + self.a_func(y) * psi - self.b_func(y) * np.log(1+np.exp(psi))
        if mask is not None:
            ll *= mask
        return ll

    def log_likelihood(self, xy, mask=None):
        ll = self._elementwise_log_likelihood(xy, mask=mask)
        return np.sum(ll, axis=1)

    @abc.abstractmethod
    def mean(self, X):
        """
        Return the expected value of y given X.
        This is distribution specific
        """
        raise NotImplementedError

    def rvs(self, x=None, size=[], return_xy=False):
        raise NotImplementedError

    def kappa_func(self, data):
        return self.a_func(data) - self.b_func(data) / 2.0

    def resample(self, data, mask=None, omega=None):
        if not isinstance(data, list):
            assert isinstance(data, tuple) and len(data) == 2, \
                "datas must be an (x,y) tuple or a list of such tuples"
            data = [data]

        if mask is None:
            mask = [np.ones(y.shape, dtype=bool) for x, y in data]

        # Resample auxiliary variables if they are not given
        if omega is None:
            omega = self._resample_auxiliary_variables(data)

        # Make copies of parameters (for sample collection in calling methods)
        self.A = self.A.copy()
        self.b = self.b.copy()

        D = self.D_in
        xs = [d[0] for d in data]
        for n in range(self.D_out):
            yns = [d[1][:,n] for d in data]
            maskns = [m[:,n] for m in mask]
            omegans = [o[:,n] for o in omega]
            self._resample_row_of_emission_matrix(n, xs, yns, maskns, omegans)

    def _resample_row_of_emission_matrix(self, n, xs, yns, maskns, omegans):
        # Resample C_{n,:} given z, omega[:,n], and kappa[:,n]
        D = self.D_in
        prior_Sigma = np.zeros((D + 1, D + 1))
        prior_Sigma[:D, :D] = self.sigmasq_A[n]
        prior_Sigma[D, D] = self.sigmasq_b[n]
        prior_J = np.linalg.inv(prior_Sigma)

        prior_h = prior_J.dot(np.concatenate((self.mu_A[n], [self.mu_b[n]])))

        lkhd_h = np.zeros(D + 1)
        lkhd_J = np.zeros((D + 1, D + 1))

        for x, yn, mn, on in zip(xs, yns, maskns, omegans):
            augx = np.hstack((x, np.ones((x.shape[0], 1))))
            Jn = on * mn
            hn = self.kappa_func(yn) * mn

            lkhd_J += (augx * Jn[:, None]).T.dot(augx)
            lkhd_h += hn.T.dot(augx)

        post_h = prior_h + lkhd_h
        post_J = prior_J + lkhd_J

        joint_sample = sample_gaussian(J=post_J, h=post_h)
        self.A[n, :] = joint_sample[:D]
        self.b[n] = joint_sample[D]

    def _resample_auxiliary_variables(self, datas):
        A, b = self.A, self.b
        omegas = []
        for data in datas:
            if isinstance(data, tuple):
                x, y = data
            else:
                x, y = data[:, :self.D_in], data[:, self.D_in:]

            bb = self.b_func(y)
            psi = x.dot(A.T) + b.T
            omega = np.zeros(y.size)
            pgdrawvpar(self.ppgs,
                       bb.ravel(),
                       psi.ravel(),
                       omega)
            omegas.append(omega.reshape(y.shape))
        return omegas

class BernoulliRegression(_PGLogisticRegressionBase):
    def a_func(self, data):
        return data

    def b_func(self, data):
        if isinstance(data, csr_matrix):
            vals = np.ones_like(data.data, dtype=np.float)
            return csr_matrix((vals, data.indices, data.indptr), shape=data.shape)
        else:
            return np.ones_like(data, dtype=np.float)

    def log_c_func(self, data):
        return 0

    def rvs(self, x=None, size=[], return_xy=False):
        if x is None:
            assert isinstance(size, int)
            x = npr.randn(size, self.D_in)

        else:
            assert x.ndim == 2 and x.shape[1] == self.D_in

        psi = x.dot(self.A.T) + self.b.T
        p = logistic(psi)
        y = npr.rand(*p.shape) < p

        return (x,y) if return_xy else y

    def mean(self, X):
        psi = X.dot(self.A.T) + self.b.T
        return logistic(psi)


    def max_likelihood(self, data, weights=None, stats=None, lmbda=0.1):
        """
        As an alternative to MCMC with Polya-gamma augmentation,
        we also implement maximum likelihood learning via gradient
        descent with autograd. This follows the pybasicbayes
        convention.

        :param data: list of tuples, (x,y), for each dataset.
        :param weights: Not used in this implementation.
        :param stats: Not used in this implementation.
        """
        import autograd.numpy as anp
        from autograd import value_and_grad, hessian_vector_product
        from scipy.optimize import minimize

        assert weights is None
        assert stats is None
        if not isinstance(data, list):
            assert isinstance(data, tuple) and len(data) == 2
            data = [data]

        # Define a helper function for the log of the logistic fn
        def loglogistic(psi):
            return psi - anp.log(1+anp.exp(psi))

        # optimize each row of A and b
        for n in range(self.D_out):

            # Define an objective function for the n-th row of hstack((A, b))
            # This is the negative log likelihood of the n-th column of data.
            def nll(abn):
                an, bn = abn[:-1], abn[-1]
                T = 0
                ll = 0
                for (x, y) in data:
                    T += x.shape[0]
                    yn = y[:, n]
                    psi = anp.dot(x, an) + bn
                    ll += anp.sum(yn * loglogistic(psi))
                    ll += anp.sum((1 - yn) * loglogistic(-1. * psi))

                # Include a penalty on the weights
                ll -= lmbda * T * anp.sum(an**2)
                ll -= lmbda * T * bn**2

                return -1 * ll / T

            abn0 = np.concatenate((self.A[n], self.b[n]))
            res = minimize(value_and_grad(nll), abn0,
                           tol=1e-3,
                           method="Newton-CG",
                           jac=True,
                           hessp=hessian_vector_product(nll))

            assert res.success
            self.A[n] = res.x[:-1]
            self.b[n] = res.x[-1]


class BinomialRegression(_PGLogisticRegressionBase):
    def __init__(self, N, D_out, D_in, **kwargs):
        self.N = N
        super(BinomialRegression, self).__init__(D_out, D_in, **kwargs)

    def a_func(self, data):
        return data

    def b_func(self, data):
        if isinstance(data, csr_matrix):
            vals = self.N * np.ones_like(data.data, dtype=np.float)
            return csr_matrix((vals, data.indices, data.indptr), shape=data.shape)
        else:
            return self.N * np.ones_like(data, dtype=np.float)

    def log_c_func(self, data):
        return gammaln(self.N+1) - gammaln(data+1) - gammaln(self.N-data+1)

    def rvs(self, x=None, size=[], return_xy=False):
        if x is None:
            assert isinstance(size, int)
            x = npr.randn(size, self.D_in)

        else:
            assert x.ndim == 2 and x.shape[1] == self.D_in

        psi = x.dot(self.A.T) + self.b.T
        p = logistic(psi)
        y = npr.binomial(self.N, p)
        return (x, y) if return_xy else y

    def mean(self, X):
        psi = X.dot(self.A.T) + self.b.T
        return self.N * logistic(psi)


class NegativeBinomialRegression(_PGLogisticRegressionBase):
    def __init__(self, r, D_out, D_in, **kwargs):
        self.r = r
        super(NegativeBinomialRegression, self).__init__(D_out, D_in, **kwargs)

    def a_func(self, data):
        return data

    def b_func(self, data):
        return self.r + data

    def log_c_func(self, data):
        return gammaln(data+self.r) - gammaln(self.r) - gammaln(data+1)

    def rvs(self, x=None, size=[], return_xy=False):
        if x is None:
            assert isinstance(size, int)
            x = npr.randn(size, self.D_in)

        else:
            assert x.ndim == 2 and x.shape[1] == self.D_in

        psi = x.dot(self.A.T) + self.b.T
        p = logistic(psi)
        y = npr.negative_binomial(self.r, 1-p)
        return (x, y) if return_xy else y

    def mean(self, X):
        psi = X.dot(self.A.T) + self.b.T
        p = logistic(psi)
        return self.r * p / (1-p)


class MultinomialRegression(_PGLogisticRegressionBase):
    def __init__(self, N, D_out, D_in, **kwargs):
        """
        Here we take D_out to be the dimension of the
        multinomial distribution's output. Once we augment,
        however, the effective dimensionality is D_out-1.
        :param N:     Number of counts in the multinomial output
        :param D_out: Number of labels in the multinomial output
        :param D_in:  Dimensionality of the inputs
        """
        self.N = N
        self.K = D_out
        assert D_out >= 2 and isinstance(D_out, int)

        # Set the mean of the offset to be standard
        # Dirichlet(alpha=1) when A=0.
        mu_b, sigmasq_b = compute_psi_cmoments(np.ones(self.K))
        default_args = dict(mu_b=mu_b, sigmasq_b=sigmasq_b)
        default_args.update(kwargs)

        # Initialize the regression as if the outputs are
        # really (D_out - 1) dimensional.
        super(MultinomialRegression, self).\
            __init__(D_out-1, D_in, **default_args)

    def a_func(self, data):
        assert data.shape[1] == self.K - 1
        return data

    def b_func(self, data):
        T = data.shape[0]
        assert data.shape[1] == self.K - 1
        return np.hstack(
            (self.N * np.ones((T,1)),
             self.N * np.ones((T, 1)) - np.cumsum(data, axis=1)[:, :-1]))

    def log_c_func(self, data):
        assert data.shape[1] == self.K - 1
        if self.N == 1:
            return 1.0
        else:
            return gammaln(self.N + 1) - \
                   np.sum(gammaln(data + 1), axis=1) - \
                   gammaln(self.N-np.sum(data, axis=1) + 1)


    def pi(self, X):
        from pypolyagamma.utils import psi_to_pi
        psi = np.dot(X, self.A.T) + self.b.T
        return psi_to_pi(psi)

    def rvs(self, x=None, size=[], return_xy=False):
        if x is None:
            assert isinstance(size, int)
            x = npr.randn(size, self.D_in)

        else:
            assert x.ndim == 2 and x.shape[1] == self.D_in

        pi = self.pi(x)
        if pi.ndim == 1:
            y = npr.multinomial(self.N, pi)
        elif pi.ndim == 2:
            y = np.array([npr.multinomial(self.N, pp) for pp in pi])
        else:
            raise NotImplementedError

        return (x, y) if return_xy else y

    def mean(self, X):
        psi = X.dot(self.A.T) + self.b.T
        return psi_to_pi(psi)

    def resample(self, data, mask=None, omega=None):
        """
        Multinomial regression is somewhat special. We have to compute the
        kappa functions for the entire dataset, not just for one column of
        the data at a time.
        """
        if not isinstance(data, list):
            assert isinstance(data, tuple) and len(data) == 2, \
                "datas must be an (x,y) tuple or a list of such tuples"
            data = [data]

        if mask is None:
            mask = [np.ones(y.shape, dtype=bool) for x, y in data]

        # Resample auxiliary variables if they are not given
        if omega is None:
            omega = self._resample_auxiliary_variables(data)

        # Make copies of parameters (for sample collection in calling methods)
        self.A = self.A.copy()
        self.b = self.b.copy()

        D = self.D_in
        for n in range(self.D_out):
            # Resample C_{n,:} given z, omega[:,n], and kappa[:,n]
            prior_Sigma = np.zeros((D + 1, D + 1))
            prior_Sigma[:D, :D] = self.sigmasq_A[n]
            prior_Sigma[D, D] = self.sigmasq_b[n]
            prior_J = np.linalg.inv(prior_Sigma)

            prior_h = prior_J.dot(np.concatenate((self.mu_A[n], [self.mu_b[n]])))

            lkhd_h = np.zeros(D + 1)
            lkhd_J = np.zeros((D + 1, D + 1))

            for d, m, o in zip(data, mask, omega):
                if isinstance(d, tuple):
                    x, y = d
                else:
                    x, y = d[:, :D], d[:, D:]
                augx = np.hstack((x, np.ones((x.shape[0], 1))))
                J = o * m
                h = self.kappa_func(y) * m

                lkhd_J += (augx * J[:, n][:, None]).T.dot(augx)
                lkhd_h += h[:, n].T.dot(augx)

            post_h = prior_h + lkhd_h
            post_J = prior_J + lkhd_J

            joint_sample = sample_gaussian(J=post_J, h=post_h)
            self.A[n, :] = joint_sample[:D]
            self.b[n] = joint_sample[D]


class _MixtureOfRegressionsBase(object):
    """
    Why have one regression when you can have a mixture?

    Technical note: we treat this as a 'Distribution' in
    the pybasicbayes sense. That is, this object does
    not maintain data-dependent state, only parameters.
    When 'resample' is called, we first instantiate latent
    variables that attribute each datapoint to one of the
    regression mixture components, then we resample each
    regression mixture component given the associated data.
    """
    _regression_class = _PGLogisticRegressionBase

    def __init__(self, M, D_out, D_in, alpha=10.0,
                 regression_kwargs={}):
        """
        Initialize the model
        :param M: Number of mixture components
        :param alpha: Concentration of mixture weights prior
        Remaining parameters are passed to regression constructor
        """
        # Instantiate M regression objects
        self.M, self.D_out, self.D_in, = M, D_out, D_in
        self.regressions = \
            [self._regression_class(D_out=D_out, D_in=D_in, **regression_kwargs)
             for _ in range(M)]

        # Weight the regression objects
        self.alpha = alpha * np.ones(M)
        # self.weights = np.ones(M) / M
        self.weights = npr.dirichlet(self.alpha)

    def mean(self, X):
        # Weighted sum of means
        means = np.array([r.mean(X) for r in self.regressions])
        return self.weights.dot(means)

    def log_likelihood(self, xy, mask=None):
        # log sum exp of lls
        lls = np.array([r.log_likelihood(xy, mask=mask) for r in self.regressions])
        lls += np.log(self.weights)[:,None]
        return logsumexp(lls, axis=0)

    def rvs(self, x=None, size=[], return_xy=False):
        # Sample mixture components
        assert x is not None, "Only supporting rvs with given covariates"

        ms = np.random.choice(self.M, p=self.weights, size=x.shape[0])
        return np.vstack([self.regressions[m].rvs(xi[None,:])
                          for m, xi in zip(ms, x)])

    def resample(self, data):
        if not isinstance(data, list):
            assert isinstance(data, tuple) and len(data) == 2, \
                "datas must be an (x,y) tuple or a list of such tuples"
            data = [data]

        # Compute the log likelihood under each mixture component and
        # assign datapoints accordingly
        zs = []
        for xy in data:
            lls = np.array([r.log_likelihood(xy) for r in self.regressions])
            from pybasicbayes.util.stats import sample_discrete_from_log
            zs.append(sample_discrete_from_log(lls, axis=0))

        # Sample each mixture component with its assigned data
        for m in range(self.M):
            data_m = [(x[z == m], y[z == m]) for (x, y), z in zip(data, zs)]
            self.regressions[m].resample(data_m)


class MixtureOfMultinomialRegressions(_MixtureOfRegressionsBase):
    """
    The multinomial regression is somewhat limited by the logistic
    stick breaking construction. By mixing multiple multinomial
    regressions (with different permutations on the output),
    we achieve a much more flexible model.
    """
    _regression_class = MultinomialRegression

    def __init__(self, M, N, D_out, D_in, Ps=None, **kwargs):
        """
        :param Ps: a set of M permutation matrices. If not given,
                   these will be randomly sampled.
        """
        kwargs["N"] = N
        super(MixtureOfMultinomialRegressions, self).\
            __init__(M, D_out, D_in, regression_kwargs=kwargs)

        # Override D_out
        self.N = N
        self.K = D_out
        self.D_out = D_out - 1

        # Initialize permutation matrices
        if Ps is None:
            Ps = []
            for m in range(M):
                perm = npr.permutation(D_out)
                P = np.zeros((D_out, D_out))
                P[np.arange(D_out), perm] = 1
                Ps.append(P)
        else:
            assert len(Ps) == M
            for P in Ps:
                assert isinstance(P, np.ndarray)
                assert P.shape == ((D_out, D_out))
                assert np.allclose(P.sum(1), 1)
                assert np.allclose(P.sum(0), 1)
                assert np.allclose((P > 0).sum(0), 1)

        self.Ps = Ps

    def log_likelihood(self, xy, mask=None):
        # TODO: Support concatenated arrays
        assert isinstance(xy, tuple)
        x,y = xy

        # If we've only been given the first K-1 dimensions of y,
        # concatenate with the last one
        assert y.ndim == 2 and x.shape[0] == y.shape[0]
        if y.shape[1] == self.K-1:
            y = np.column_stack((y, self.N-y.sum(1)))

        # log sum exp of lls
        lls = []
        for m in range(self.M):
            # Undo the permutation
            # By definition, y_t ~ P pi_t where pi_t is a column vector
            # equivalently, y ~ pi P^T when y and pi are T x K (i.e. y_t and pi_t are row vectors)
            # To undo the permutation, right multiply by P = (P^T)^{-1}
            y_unperm = y.dot(self.Ps[m])
            ll_m = self.regressions[m].log_likelihood((x, y_unperm[:,:-1]), mask=mask)
            ll_m += np.log(self.weights[m])
            lls.append(ll_m)

        return logsumexp(np.array(lls), axis=0)

    def pi(self, X):
        # Compute probabilities for each multinomial regression
        pis = np.array([r.pi(X) for r in self.regressions])

        # Permute the probabilities according to p
        pis = np.array([pi.dot(P.T) for pi, P in zip(pis, self.Ps)])

        # Weight the probabilities
        pis = np.tensordot(self.weights, pis, axes=([0], [0]))

        return pis

    def rvs(self, x=None, size=[], return_xy=False):
        # Sample mixture components
        assert x is not None, "Only supporting rvs with given covariates"

        ms = np.random.choice(self.M, p=self.weights, size=x.shape[0])
        return np.vstack([self.regressions[m].rvs(xi[None,:]).dot(self.Ps[m].T)
                          for m, xi in zip(ms, x)])


    def resample(self, data):
        if not isinstance(data, list):
            assert isinstance(data, tuple) and len(data) == 2, \
                "datas must be an (x,y) tuple or a list of such tuples"
            data = [data]

        # Compute the log likelihood under each mixture component and
        # assign datapoints accordingly
        zs = []
        for x,y in data:
            # If we've only been given the first K-1 dimensions of y,
            # concatenate with the last one
            assert y.ndim == 2 and x.shape[0] == y.shape[0]
            if y.shape[1] == self.K - 1:
                y = np.column_stack((y, self.N - y.sum(1)))

            lls = []
            for m in range(self.M):
                # Undo the permutation
                # By definition, y_t ~ P pi_t where pi_t is a column vector
                # equivalently, y ~ pi P^T when y and pi are T x K (i.e. y_t and pi_t are row vectors)
                # To undo the permutation, right multiply by P = (P^T)^{-1}
                y_unperm = y.dot(self.Ps[m])
                ll_m = self.regressions[m].log_likelihood((x, y_unperm[:, :-1]))
                ll_m += np.log(self.weights[m])
                lls.append(ll_m)
            lls = np.array(lls)

            from pybasicbayes.util.stats import sample_discrete_from_log
            zs.append(sample_discrete_from_log(lls, axis=0))

        # Sample each mixture component with its assigned data
        for m in range(self.M):
            data_m = [(x[z==m], y[z==m]) for (x,y),z in zip(data, zs)]
            self.regressions[m].resample(data_m)

        # Sample mixture weights
        alpha_hat = self.alpha + sum([np.bincount(z, minlength=self.M) for z in zs])
        self.weights = npr.dirichlet(alpha_hat)
