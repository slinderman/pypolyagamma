from __future__ import absolute_import
from warnings import warn
from pypolyagamma.pypolyagamma import PyRNG, PyPolyaGamma

# Try to import the parallel version, but if they didn't compile,
# revert to the serial versions
try:
    from pypolyagamma.parallel import pgdrawvpar, get_omp_num_threads

except:
    def get_omp_num_threads():
        return 1

    def pgdrawvpar(ppgs, ns, zs, pgs):
        warn("PyPolyaGamma was not installed with OpenMP. Calls to 'pgdrawvpar' will "
             "be replaced with a serial implementation.")

        assert isinstance(ppgs, list)
        ppg = ppgs[0]
        ppg.pgdrawv(ns, zs, pgs)

from pypolyagamma.utils import pgpdf, pgmean, logistic
from pypolyagamma.distributions import BernoulliRegression, \
    BinomialRegression, NegativeBinomialRegression, MultinomialRegression, \
    MixtureOfMultinomialRegressions

