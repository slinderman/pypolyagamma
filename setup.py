import os

from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext as _build_ext
import pkg_resources

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


# Manually define the list of sources, including GSL files
include_dirs = \
    [
        "pypolyagamma/cpp/include",
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
    ]

libraries = ["gsl", "gslcblas"]


# Create the extensions. Manually enumerate the required
extensions = []

# PyPolyaGamma and GSL source files
extensions.append(
    Extension('pypolyagamma.pypolyagamma',
              depends=headers,
              extra_compile_args=[],
              extra_link_args=[],
              include_dirs=include_dirs,
              libraries=libraries,
              language="c++",
              sources=["pypolyagamma/pypolyagamma" + ext] + sources,
              )
)

# If OpenMP is requested, compile the parallel extension
if USE_OPENMP:
    extensions.append(
        Extension('pypolyagamma.parallel',
                  depends=headers,
                  extra_compile_args=["-fopenmp"],
                  extra_link_args=["-fopenmp"],
                  include_dirs=include_dirs,
                  libraries=libraries,
                  language="c++",
                  sources=["pypolyagamma/parallel" + ext] + sources,
                  )
    )

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

setup(
    name='pypolyagamma',
    version='1.1.7',
    description='''Cython wrappers for Polya gamma random number generation based on Jesse Windle\'s BayesLogit package: https://github.com/jwindle/BayesLogit.''',
    author='Scott Linderman',
    author_email='scott.linderman@columbia.edu',
    url='http://www.github.com/slinderman/pypolyagamma',
    license="MIT",
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
