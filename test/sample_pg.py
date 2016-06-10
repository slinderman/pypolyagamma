"""
Call the different sample methods
"""
from __future__ import print_function
import numpy as np
np.random.seed(0)
import pypolyagamma as pypolyagamma

rng = pypolyagamma.PyRNG(0)
ppg = pypolyagamma.PyPolyaGamma(np.random.randint(2**16))

# # Call the single sample
# v1 = ppg.pgdraw(1.,1.)
# print v1
#
# # Call the vectorized version
n = 5
# v2 = np.zeros(n)
a = 14*np.ones(n, dtype=np.float)
b = 0*np.ones(n, dtype=np.float)
# ppg.pgdrawv(a, b, v2)
# print v2

# Call the parallel vectorized version
# n = 5
nthreads = 8
v3 = np.zeros(n)
seeds = np.random.randint(2**16, size=nthreads)
ppgs = [pypolyagamma.PyPolyaGamma(seed) for seed in seeds]
pypolyagamma.pgdrawvpar(ppgs, a, b, v3)
print(v3)
