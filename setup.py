import os

from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext as _build_ext
import pkg_resources


from glob import glob
import tarfile
import shutil
import subprocess

# Not the greatest way to handle Python 2/3 changes
# but this avoids the need for the 'future' modules,
# which was causing some headaches in installation.
try:
    # Python 2
    from urllib import urlretrieve
except:
    try:
        # Python 3
        from urllib.request import urlretrieve
    except:
        raise Exception("Could not import urlretrieve.")


# Hold off on locating the numpy include directory 
# until we are actually building the extensions, by which 
# point numpy should have been installed
class build_ext(_build_ext):
    def build_extensions(self):
        numpy_incl = pkg_resources.resource_filename('numpy', 'core/include')
        for ext in self.extensions:
            if hasattr(ext, 'include_dirs') and not numpy_incl in ext.include_dirs:
                ext.include_dirs.append(numpy_incl)
        _build_ext.build_extensions(self)

# Dealing with Cython
# use cython if we can import it successfully
try:
    from Cython.Distutils import build_ext as _build_ext
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

# Only compile with OpenMP if user asks for it
USE_OPENMP = os.environ.get('USE_OPENMP', False)

#  If not using Cython, make sure the cpp files are present
ext = ".pyx" if USE_CYTHON else ".cpp"
if not USE_CYTHON:
    # Make sure that the CPP files are present
    assert os.path.exists(os.path.join("pypolyagamma", "pypolyagamma.cpp"))
    if USE_OPENMP:
        assert os.path.exists(os.path.join("pypolyagamma", "parallel.cpp"))

# download GSL if we don't have it in deps
if not os.path.exists('deps'):
    os.makedirs('deps')
gslurl = 'http://ftp.snt.utwente.nl/pub/software/gnu/gsl/gsl-latest.tar.gz'
gsltarpath = os.path.join('deps', 'gsl-latest.tar.gz')
gslpath = os.path.join('deps', 'gsl')
if not os.path.exists(gslpath):
    print('Downloading GSL...')
    urlretrieve(gslurl, gsltarpath)
    print("Extracting to {}".format(gslpath))
    with tarfile.open(gsltarpath, 'r') as tar:
        tar.extractall('deps')
    thedir = glob(os.path.join('deps', 'gsl-*/'))[0]
    shutil.copytree(os.path.join(thedir), gslpath)
    print('...Done!')

# Check if GSL has been configured
if not os.path.exists(os.path.join(gslpath, "config.h")):
    # Run configure to make config.h
    subprocess.call("./configure", cwd=gslpath, shell=True)

# Check if the GSL headers have been symlinked
if not os.path.exists(os.path.join(gslpath, "gsl", "gsl_rng.h")):
    # Run make to symlink the headers
    subprocess.call("make", cwd=os.path.join(gslpath, "gsl"), shell=True)


# Manually define the list of sources, including GSL files
include_dirs = \
    [
        "pypolyagamma/cpp/include",
        "deps/gsl",
        "deps/gsl/gsl",
    ]

headers = \
    [
        "pypolyagamma/cpp/PolyaGammaHybrid.h",
        "pypolyagamma/cpp/include/RNG.hpp"
    ]

sources = \
    [
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
        "deps/gsl/err/stream.c"
    ]

# Create the extensions. Manually enumerate the required
extensions = []

# PyPolyaGamma and GSL source files
extensions.append(
    Extension('pypolyagamma.pypolyagamma',
              depends=headers,
              extra_compile_args=["-w", "-DHAVE_INLINE"],
              extra_link_args=[],
              include_dirs=include_dirs,
              language="c++",
              sources=["pypolyagamma/pypolyagamma" + ext] + sources,
              )
)

# If OpenMP is requested, compile the parallel extension
if USE_OPENMP:
    extensions.append(
        Extension('pypolyagamma.parallel',
                  depends=headers,
                  extra_compile_args=["-w","-fopenmp", "-DHAVE_INLINE"],
                  extra_link_args=["-fopenmp"],
                  include_dirs=include_dirs,
                  language="c++",
                  sources=["pypolyagamma/parallel" + ext] + sources,
                  )
    )

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

setup(
    name='pypolyagamma',
    version='1.2.2',
    description='''Cython wrappers for Polya gamma random number generation based on Jesse Windle\'s BayesLogit package: https://github.com/jwindle/BayesLogit.''',
    author='Scott Linderman',
    author_email='scott.linderman@columbia.edu',
    url='http://www.github.com/slinderman/pypolyagamma',
    license="GNU GPLv3",
    packages=['pypolyagamma'],
    ext_modules=extensions,
    install_requires=['numpy', 'scipy', 'matplotlib'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: C++',
        ],
    keywords=['monte-carlo', 'polya', 'gamma'],
    platforms="ALL",
    cmd_class = {'build_ext': build_ext}
)
