"""
Call the different sample methods
"""
import numpy as np
import pypolyagamma as ppg

rng = ppg.PyRNG()

# Call the single sample
v1 = ppg.pgdraw(1,1,rng)
print v1

# Call the vectorized version
n = 100
v2 = np.zeros(n)
a = np.ones(n, dtype=np.int32)
b = np.ones(n, dtype=np.float)
ppg.pgdrawv(a, b, v2, rng)
print v2

# Call the parallel vectorized version
n = 100
nthreads = 10
v3 = np.zeros(n)
ppg.pgdrawv_par(a, b, v3, nthreads)
print v3