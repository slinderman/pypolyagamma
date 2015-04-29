"""
Call the different sample methods
"""
import numpy as np
np.random.seed(0)
import pypolyagamma as ppg

# Call the parallel vectorized version
n = 10000
b = 160*np.ones(n, dtype=np.float)
z = -87*np.ones(n, dtype=np.float)
v3 = np.zeros(n)
#
# print "Different seeds"
# nthreads = 1
# seeds = np.random.randint(2**16, size=nthreads)
# ppgs = [ppg.PyPolyaGamma(seed) for seed in seeds]
# ppg.pgdrawvpar(ppgs, b, z, v3)
# print v3

# Now try it where they all have the same seed
print "Same seed"
nthreads = ppg.get_omp_num_threads()
print "N threads: ", nthreads
seeds = np.zeros(nthreads, dtype=np.uint)
ppgs = [ppg.PyPolyaGamma(seed) for seed in seeds]
ppg.pgdrawvpar(ppgs, b, z, v3)
print v3
