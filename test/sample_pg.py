"""
Call the different sample methods
"""
from __future__ import print_function
import numpy as np
import pypolyagamma as pypolyagamma

# No seed
def test_no_seed():
    ppg = pypolyagamma.PyPolyaGamma()
    v1 = ppg.pgdraw(1., 1.)
    print(v1)

# Call the single sample
def test_single_draw():
    np.random.seed(0)
    ppg = pypolyagamma.PyPolyaGamma(np.random.randint(2 ** 16))
    v1 = ppg.pgdraw(1., 1.)
    print(v1)

# Sample a vector
def test_vector_draw():
    np.random.seed(0)
    ppg = pypolyagamma.PyPolyaGamma(np.random.randint(2 ** 16))

    # Call the vectorized version
    n = 5
    v2 = np.zeros(n)
    a = 14*np.ones(n, dtype=np.float)
    b = 0*np.ones(n, dtype=np.float)
    ppg.pgdrawv(a, b, v2)
    print(v2)


def test_parallel():
    # Call the parallel vectorized version
    np.random.seed(0)

    n = 5
    nthreads = 8
    v3 = np.zeros(n)
    a = 14 * np.ones(n)
    b = 0 * np.ones(n)
    seeds = np.random.randint(2**16, size=nthreads)
    ppgs = [pypolyagamma.PyPolyaGamma(seed) for seed in seeds]
    pypolyagamma.pgdrawvpar(ppgs, a, b, v3)
    print(v3)


if __name__ == "__main__":
    test_no_seed()
    test_single_draw()
    test_vector_draw()
    test_parallel()