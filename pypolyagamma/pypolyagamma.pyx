# distutils: language = c++
# distutils: sources = pypolyagamma/cpp/PolyaGamma.cpp pypolyagamma/cpp/PolyaGammaAlt.cpp pypolyagamma/cpp/PolyaGammaSP.cpp pypolyagamma/cpp/InvertY.cpp pypolyagamma/cpp/include/RNG.cpp pypolyagamma/cpp/include/GRNG.cpp
# distutils: libraries = stdc++ gsl gslcblas
# distutils: library_dirs = /usr/local/lib
# distutils: include_dirs = pypolyagamma/cpp/include /usr/local/include
# distutils: extra_compile_args = -O0 -w -std=c++11 -fopenmp
# distutils: extra_link_args = -fopenmp
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

from libc.stdio cimport printf

from cython.parallel import prange, parallel
from libcpp.vector cimport vector

from cython.operator cimport dereference as deref
from openmp cimport omp_get_num_threads, omp_get_thread_num, omp_get_max_threads

import numpy as np
cimport numpy as np
from scipy.special import gammaln
from scipy.misc import logsumexp

# Import C++ classes from RNG.h
cdef extern from "cpp/include/RNG.hpp":

    # RNG class
    cdef cppclass RNG:
        RNG(unsigned long seed) except +


# Import C++ classes from PolyaGammaHybrid.h
cdef extern from "cpp/PolyaGammaHybrid.h":

    # PolyaGammaHybrid class
    cdef cppclass PolyaGammaHybridDouble:
        PolyaGammaHybridDouble(unsigned long seed) except +
        double draw(double b_, double z_) nogil except +
        void set_trunc(int trunc) except +


# Expose the RNG class to Python
cdef class PyRNG:
    cdef RNG *thisptr

    def __cinit__(self, unsigned long seed=0):
        self.thisptr = new RNG(seed)

    def __dealloc__(self):
        del self.thisptr

# Expose the RNG class to Python
cdef class PyPolyaGamma:
    cdef PolyaGammaHybridDouble *thisptr

    def __cinit__(self, unsigned long seed=0, int trunc=40):
        self.thisptr = new PolyaGammaHybridDouble(seed)
        self.set_trunc(trunc)

    def __dealloc__(self):
        del self.thisptr

    cpdef set_trunc(self, int trunc):
        self.thisptr.set_trunc(trunc)

    cpdef double pgdraw(self, double n, double z):
        return self.thisptr.draw(n, z)

    cpdef pgdrawv(self, double[::1] ns, double[::1] zs, double[::1] pgs):
        """
        Draw a vector of Polya-gamma random variables
        """

        cdef int s = 0
        cdef int S = ns.size

        for s in range(S):
            pgs[s] = self.thisptr.draw(ns[s], zs[s])

cpdef pgdrawvpar(list ppgs, double[::1] ns, double[::1] zs, double[::1] pgs):
    """
    Draw a vector of Polya-gamma random variables in parallel
    """
    # Make a vector of PyPolyaGammDouble C++ objects
    cdef int m = len(ppgs)
    cdef vector[PolyaGammaHybridDouble*] ppgsv
    for ppg in ppgs:
        ppgsv.push_back((<PyPolyaGamma>ppg).thisptr)

    cdef int s = 0
    cdef int S = ns.size

    cdef int num_threads, blocklen, sequence_idx_start, sequence_idx_end, thread_num

    with nogil, parallel():
        # Fix up assignments to avoid cache collisions
        num_threads = omp_get_num_threads()
        thread_num = omp_get_thread_num()
        blocklen = 1 + ((S - 1) / num_threads)
        sequence_idx_start = blocklen * thread_num
        sequence_idx_end = min(sequence_idx_start + blocklen, S)

        # TODO: Make sure there is a ppg sampler for each thread
        for s in range(sequence_idx_start, sequence_idx_end):
            pgs[s] = ppgsv[thread_num].draw(ns[s], zs[s])


cpdef silly_pgdrawvpar(list ppgs, double[::1] ns, double[::1] zs, double[::1] pgs):
    """
    Draw a vector of Polya-gamma random variables in parallel
    """
    # Make a vector of PyPolyaGammDouble C++ objects
    cdef int m = len(ppgs)
    cdef vector[PolyaGammaHybridDouble*] ppgsv
    for ppg in ppgs:
        ppgsv.push_back((<PyPolyaGamma>ppg).thisptr)

    cdef int s = 0
    cdef int S = ns.size

    with nogil:
        for s in prange(S):
            pgs[s] = ppgsv[s % m].draw(ns[s], zs[s])

cpdef int get_omp_num_threads():
    # This might not be kosher
    cdef int num_threads = omp_get_max_threads()
    return num_threads

def pgpdf(omega, b, psi, trunc=200):
    """
    Approximate the density log PG(omega | b, psi) using a
    truncation of the density written as an infinite sum.

    :param omega: point at which to evaluate density
    :param b:   first parameter of PG
    :param psi: tilting of PG
    :param trunc: number of terms in sum
    """
    logZ = b * np.log(np.cosh(psi/2.)) + (b-1) * np.log(2) -gammaln(b)
    lp =  0
    nlp = 0
    for n in xrange(trunc):
        sign = -1 ** n
        trm = gammaln(n+b) - gammaln(n+1)
        trm += np.log(2*n + b) - 0.5 * np.log(2*np.pi*omega**3)
        trm += -(2*n+b)**2 / (8*omega) - psi**2/2. * omega

        if n % 2 == 0:
            lp += trm
        else:
            nlp += trm

    # Compute the sum of these terms
    sumlp = logsumexp(lp)
    sumnlp = logsumexp(nlp)
    p = np.exp(lp + sumlp) - np.exp(sumnlp)
    return p

def pgmean(b, psi):
    return b / (2.*psi) * np.tanh(psi/2.)