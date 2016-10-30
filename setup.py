import os

from setuptools import setup
from setuptools.extension import Extension

try:
    import numpy as np
except ImportError:
    print("Please install numpy.")

# Dealing with Cython
# USE_CYTHON = os.environ.get('USE_CYTHON', False)
# ext = '.pyx' if USE_CYTHON else '.cpp'
#
# extensions = [
#     Extension('pypolyagamma.pypolyagamma',
#               ['pypolyagamma/pypolyagamma' + ext],
#               include_dirs=[np.get_include(),
#                             'pypolyagamma/cpp',
#                             'pypolyagamma/cpp/include'])
# ]
#
# if USE_CYTHON:
#     from Cython.Build import cythonize
#     extensions = cythonize(extensions)
from Cython.Build import cythonize
extensions = cythonize('**/*.pyx')

setup(
    name='pypolyagamma',
    version='0.2.1',
    description='''Cython wrappers for Polya gamma random number generation based on Jesse
                   Windle\'s BayesLogit package: https://github.com/jwindle/BayesLogit.''',
    author='Scott Linderman',
    author_email='slinderman@seas.harvard.edu',
    url='http://www.github.com/slinderman/pypolyagamma',
    license="MIT",
    packages=['pypolyagamma'],
    ext_modules=extensions,
    include_dirs=[np.get_include()],
    install_requires=[
        'numpy',
        ],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: C++',
        ],
    keywords=['monte-carlo', 'polya', 'gamma'],
    platforms="ALL",
)
