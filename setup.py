import os

from setuptools import setup
from setuptools.extension import Extension

from glob import glob
from future.moves.urllib.request import urlretrieve
import tarfile
import shutil

import subprocess

try:
    import numpy as np
except ImportError:
    print("Please install numpy.")

# Dealing with Cython
USE_CYTHON = os.environ.get('USE_CYTHON', False)
ext = '.pyx' if USE_CYTHON else '.cpp'

# download GSL if we don't have it in deps
gslurl = 'http://open-source-box.org/gsl/gsl-latest.tar.gz'
gsltarpath = os.path.join('deps', 'gsl-latest.tar.gz')
gslpath = os.path.join('deps', 'gsl')
if not os.path.exists(gslpath):
    print('Downloading GSL...')
    urlretrieve(gslurl, gsltarpath)
    print("Extracting to {}".format(gslpath))
    with tarfile.open(gsltarpath, 'r') as tar:
        tar.extractall('deps')
    thedir = glob(os.path.join('deps', 'gsl-*'))[0]
    shutil.move(os.path.join(thedir), gslpath)
    print('...Done!')

# Run configure to copy headers to expected locations
subprocess.run("./configure", cwd=gslpath, shell=True)

# Create the extensions. Manually enumerate the required
# PyPolyaGamma and GSL source files
extensions = [
    Extension('pypolyagamma.pypolyagamma',
              depends=[
                  "pypolyagamma/cpp/PolyaGammaHybrid.h",
                  "pypolyagamma/cpp/include/RNG.hpp"],
              extra_compile_args=["-w","-fopenmp", "-DHAVE_INLINE"],
              extra_link_args=["-fopenmp"],
              include_dirs=[
                  "pypolyagamma/cpp/include",
                  "deps/gsl",
                  "deps/gsl/gsl",
                  np.get_include()],
              language="c++",
              sources=[
                   "pypolyagamma/pypolyagamma" + ext,
                   "pypolyagamma/cpp/PolyaGamma.cpp",
                   "pypolyagamma/cpp/PolyaGammaSmallB.cpp",
                   "pypolyagamma/cpp/PolyaGammaAlt.cpp",
                   "pypolyagamma/cpp/PolyaGammaSP.cpp",
                   "pypolyagamma/cpp/InvertY.cpp",
                   "pypolyagamma/cpp/include/RNG.cpp",
                   "pypolyagamma/cpp/include/GRNG.cpp",
                   "deps/gsl/rng/mt.c",
                   "deps/gsl/cdf/gamma.c",
                   "deps/gsl/cdf/gauss.c",
                   "deps/gsl/randist/bernoulli.c",
                   "deps/gsl/randist/beta.c",
                   "deps/gsl/randist/chisq.c",
                   "deps/gsl/randist/exponential.c",
                   "deps/gsl/randist/flat.c",
                   "deps/gsl/randist/gamma.c",
                   "deps/gsl/randist/gauss.c",
                   "deps/gsl/randist/gausszig.c",
                   "deps/gsl/rng/rng.c",
                   "deps/gsl/err/error.c",
                   "deps/gsl/rng/file.c",
                   "deps/gsl/specfunc/gamma.c",
                   "deps/gsl/specfunc/gamma_inc.c",
                   "deps/gsl/specfunc/erfc.c",
                   "deps/gsl/specfunc/exp.c",
                   "deps/gsl/specfunc/expint.c",
                   "deps/gsl/specfunc/trig.c",
                   "deps/gsl/specfunc/log.c",
                   "deps/gsl/specfunc/psi.c",
                   "deps/gsl/specfunc/zeta.c",
                   "deps/gsl/specfunc/elementary.c",
                   "deps/gsl/complex/math.c",
                   "deps/gsl/sys/infnan.c",
                   "deps/gsl/sys/fdiv.c",
                   "deps/gsl/sys/coerce.c",
                   "deps/gsl/err/stream.c"],
              )
]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

setup(
    name='pypolyagamma',
    version='0.3.1',
    description='''Cython wrappers for Polya gamma random number generation based on Jesse
                   Windle\'s BayesLogit package: https://github.com/jwindle/BayesLogit.''',
    author='Scott Linderman',
    author_email='slinderman@seas.harvard.edu',
    url='http://www.github.com/slinderman/pypolyagamma',
    license="MIT",
    packages=['pypolyagamma'],
    ext_modules=extensions,
    install_requires=['numpy',],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: C++',
        ],
    keywords=['monte-carlo', 'polya', 'gamma'],
    platforms="ALL",
)
