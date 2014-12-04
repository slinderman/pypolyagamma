# distutils: language = c++
# distutils: sources = cpp/PolyaGamma.cpp cpp/include/RNG.cpp cpp/include/GRNG.cpp
# distutils: libraries = stdc++ gsl gslcblas
# distutils: library_dirs = /usr/local/lib
# distutils: include_dirs = cpp/include /usr/local/include
# distutils: extra_compile_args = -O3 -w -std=c++11
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

from cython.parallel import prange
from libcpp.vector cimport vector


# Import C++ classes from RNG.h
cdef extern from "cpp/include/RNG.hpp":

    # RNG class
    cdef cppclass RNG:
        RNG() except +

# Expose the RNG class to Python
cdef class PyRNG:
    cdef RNG *thisptr

    def __cinit__(self):
        self.thisptr = new RNG()

    def __dealloc__(self):
        del self.thisptr

# Expose the PolyaGamma class to Python
cdef extern from "cpp/PolyaGamma.h":
    double draw(int n, double z, RNG* rng) nogil

cpdef double pgdraw(int n, double z, PyRNG rng):
    return draw(n, z, rng.thisptr)

cpdef pgdrawv(int[::1] ns, double[::1] zs, double[::1] pgs, PyRNG rng):
    """
    Draw a vector of Polya-gamma random variables
    """

    cdef int s = 0
    cdef int S = ns.size

    for s in range(S):
        pgs[s] = draw(ns[s], zs[s], rng.thisptr)


cpdef pgdrawv_par(int[::1] ns, double[::1] zs, double[::1] pgs, int nthreads):
    """
    Draw a vector of Polya-gamma random variables
    """
    cdef int s = 0
    cdef int S = ns.size


    cdef vector[RNG*] rngs
    cdef int i
    for i in range(nthreads):
        rngs.push_back(new RNG())

    with nogil:
        for s in prange(S, num_threads=nthreads):
            pgs[s] = draw(ns[s], zs[s], rngs[s % nthreads])
