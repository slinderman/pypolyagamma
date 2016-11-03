# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

# Fill in the RNG class to Python
cdef class PyRNG:
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

    cpdef double pgdraw(self, double b, double c):
        return self.thisptr.draw(b, c)

    cpdef pgdrawv(self, double[::1] bs, double[::1] cs, double[::1] pgs):
        """
        Draw a vector of Polya-gamma random variables
        """

        cdef int s = 0
        cdef int S = bs.size

        for s in range(S):
            pgs[s] = self.thisptr.draw(bs[s], cs[s])
