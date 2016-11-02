# PyPólyaGamma

This is a Cython port of Jesse Windle's code at
https://github.com/jwindle/BayesLogit. It provides a
Python interface for efficiently sampling Pólya-gamma
random variates.

# Demo

`TODO`

# Installation

This is the simplest way to install:

    git clone git@github.com:slinderman/pypolyagamma.git
    cd pypolyagamma
    pip install -e .

Under the hood, the intaller will download
[GSL](https://www.gnu.org/software/gsl/),
untar it, and place it in `deps/gsl`. It will then configure GSL and
compile the Pólya-gamma code along with the required GSL source files.
This way, you don't need GSL to be installed and available on your
library path. 

## Parallel sampling with OpenMP
By default, the simple installation above will not support
parallel sampling. If you are compiling with GNU `gcc` and `g++`,
you can enable OpenMP support with the flag:

    USE_OPENMP=True pip install -e .

Mac users: you can install `gcc` and `g++` with Homebrew. Just
make sure that they are your default compilers, e.g. by setting
the environment variables `CC` and `CXX` to point to the GNU versions
of `gcc` and `g++`, respectively. With Homebrew, these versions
will be in `/usr/local/bin` by default.

To sample in parallel, call the `pgdrawvpar` method:

    n = 10             # Number of variates to sample
    b = np.ones(n)     # Vector of shape parameters
    c = np.zeros(n)    # Vector of tilting parameters
    out = np.empty(n)  # Outputs

    # Construct a set of PolyaGamma objects for sampling
    nthreads = 8
    seeds = np.random.randint(2**16, size=nthreads)
    ppgs = [pypolyagamma.PyPolyaGamma(seed) for seed in seeds]

    # Sample in parallel
    pypolyagamma.pgdrawvpar(ppgs, b, c, out)

If you haven't installed with OpenMP, this function will
revert to the serial sampler. 