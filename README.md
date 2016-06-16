This is a simple Cython port of Jesse Windle's code at
https://github.com/jwindle/BayesLogit.

Try this: make sure you have gsl installed (`brew install gsl`) and that you
are using gcc (not clang) (`export CC=gcc CXX=g++`), then run `pip install
pypolyagamma`.

Import in python with `import pypolyagamma`.

To build and install from source, run `python setup.py build_ext --inplace`, To
test, run `python test/sample_pg.py`

Some may also need to `export
DYLD_LIBRARY_PATH=/usr/local/Cellar/gcc/6.1.0/lib/gcc/6/`, you may have to
point somewhere different depending on how you installed `gcc`. This example
used `brew install gcc`.

Tested on Mac OSX Yosemite with gcc 4.9, 5, and 6 (not clang)
