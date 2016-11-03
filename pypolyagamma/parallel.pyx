# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True

from cython.parallel import prange, parallel
from libcpp.vector cimport vector

from openmp cimport omp_get_num_threads, omp_get_thread_num, omp_get_max_threads

cimport pypolyagamma
import pypolyagamma

cpdef pgdrawvpar(list ppgs, double[::1] ns, double[::1] zs, double[::1] pgs):
    """
    Draw a vector of Polya-gamma random variables in parallel
    """
    # Make a vector of PyPolyaGammDouble C++ objects
    cdef int m = len(ppgs)
    cdef vector[pypolyagamma.PolyaGammaHybridDouble*] ppgsv
    for ppg in ppgs:
        ppgsv.push_back((<pypolyagamma.PyPolyaGamma>ppg).thisptr)

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

cpdef int get_omp_num_threads():
    # This might not be kosher
    cdef int num_threads = omp_get_max_threads()
    return num_threads
