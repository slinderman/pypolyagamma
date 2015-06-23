This is a simple Cython port of Jesse Windle's code at https://github.com/jwindle/BayesLogit.

Try this: make sure you have gsl installed (I put it in `/usr/local/lib`) and that you are using 
gcc (`export CXX=gcc`), then run `pip intall pypolyagamma`.

To build and install from source, run `python setup.py build_ext --inplace`, 
To test,  run `ipython test/sample_pg.py`

Tested on Mac OSX Yosemite with gcc 4.9 (not clang) 
