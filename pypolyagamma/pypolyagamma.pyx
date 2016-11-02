# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

# Fill in the RNG class to Python
cdef class PyRNG:
    # cdef RNG *thisptr

    def __cinit__(self, unsigned long seed=0):
        self.thisptr = new RNG(seed)

    def __dealloc__(self):
        del self.thisptr

# Fill in PyPolyaGamma calss
cdef class PyPolyaGamma:
    def __cinit__(self, unsigned long seed=0, int trunc=200):
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
