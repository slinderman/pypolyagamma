# distutils: language = c++
# distutils: sources = cpp/PolyaGamma.cpp cpp/PolyaGammaAlt.cpp cpp/PolyaGammaSP.cpp cpp/InvertY.cpp cpp/include/RNG.cpp cpp/include/GRNG.cpp
# distutils: libraries = stdc++ gsl gslcblas
# distutils: library_dirs = /usr/local/lib
# distutils: include_dirs = cpp/include /usr/local/include
# distutils: extra_compile_args = -O3 -w -std=c++11
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

from cython.parallel import prange
from libcpp.vector cimport vector

from cython.operator cimport dereference as deref


# Import C++ classes from RNG.h
cdef extern from "cpp/include/RNG.hpp":

    # RNG class
    cdef cppclass RNG:
        RNG() except +


# Import C++ classes from PolyaGammaHybrid.h
cdef extern from "cpp/PolyaGammaHybrid.h":

    # PolyaGammaHybrid class
    cdef cppclass PolyaGammaHybridDouble:
        PolyaGammaHybridDouble() except +
        double draw(double b_, double z_, RNG& r) except +


# Expose the RNG class to Python
cdef class PyRNG:
    cdef RNG *thisptr

    def __cinit__(self):
        self.thisptr = new RNG()

    def __dealloc__(self):
        del self.thisptr

# Expose the RNG class to Python
cdef class PyPolyaGamma:
    cdef PolyaGammaHybridDouble *thisptr

    def __cinit__(self):
        self.thisptr = new PolyaGammaHybridDouble()

    def __dealloc__(self):
        del self.thisptr

    cpdef double pgdraw(self, double n, double z, PyRNG rng):
        return self.thisptr.draw(n, z, deref(rng.thisptr))

    cpdef pgdrawv(self, double[::1] ns, double[::1] zs, double[::1] pgs, PyRNG rng):
        """
        Draw a vector of Polya-gamma random variables
        """

        cdef int s = 0
        cdef int S = ns.size

        for s in range(S):
            pgs[s] = self.thisptr.draw(ns[s], zs[s], deref(rng.thisptr))

