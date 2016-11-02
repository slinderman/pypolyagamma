# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

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


# Expose the RNG class to Python
cdef class PyPolyaGamma:
    cdef PolyaGammaHybridDouble *thisptr
    cpdef set_trunc(self, int trunc)
    cpdef double pgdraw(self, double n, double z)
    cpdef pgdrawv(self, double[::1] ns, double[::1] zs, double[::1] pgs)
