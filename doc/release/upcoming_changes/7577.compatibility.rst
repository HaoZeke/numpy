``f2py`` call-back functions return complex results by value
------------------------------------------------------------
Call-back functions that ``f2py`` generates for ``COMPLEX`` functions now return the result by value, as GNU Fortran and LLVM Flang expect.
With the hidden-pointer signature, a call from those compilers crashed or returned the wrong complex value (gh-7577).
Intel Fortran, NVIDIA HPC Fortran, AMD AOCC Flang, and ``gfortran -ff2c`` pass the address of the result as a hidden first argument.
The ``meson`` backend of ``f2py -c`` passes ``-DF2PY_CB_RETURNCOMPLEX=0`` for the Intel and NVIDIA compilers.
Any other build that uses a compiler with a hidden first argument needs that flag in the C compiler flags.
