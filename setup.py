from distutils.core import setup
from Cython.Build import cythonize

import numpy as np

setup(
    name='pypolyagamma',
    version='0.2',
    description='Cython wrappers for Polya gamma random number generation based on Jesse Windle\'s BayesLogit package: https://github.com/jwindle/BayesLogit',
    author='Scott Linderman',
    author_email='slinderman@seas.harvard.edu',
    url='http://www.github.com/slinderman/pypolyagamma',
    license="MIT",
    packages=['pypolyagamma'],
    ext_modules=cythonize('**/*.pyx'),
    include_dirs=[np.get_include(),],
    install_requires=[
        'Cython >= 0.20.1',
        'numpy'
        ],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: C++',
        ],
    keywords=['monte-carlo', 'polya', 'gamma'],
    platforms="ALL",
)

