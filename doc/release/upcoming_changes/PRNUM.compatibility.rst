``f2py`` call-back functions return complex results by value
-------------------------------------------------------------
Call-back functions that ``f2py`` generates for ``COMPLEX`` functions now
return the result by value, as GNU Fortran and LLVM Flang expect. Previously
they took the address of the result as a hidden first argument unless the C
code was compiled with ``-DF2PY_CB_RETURNCOMPLEX``, so calling a Python
function that returns a complex value from Fortran crashed or produced wrong
results with these compilers. Intel Fortran, NVIDIA HPC Fortran, AMD AOCC Flang
and ``gfortran -ff2c`` use the hidden argument: the ``meson`` backend of
``f2py -c`` now passes ``-DF2PY_CB_RETURNCOMPLEX=0`` for the Intel and NVIDIA
compilers, and any other build that uses such a compiler needs to add it to
the C compiler flags.
