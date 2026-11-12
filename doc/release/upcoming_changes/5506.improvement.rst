``f2py`` accepts ``**`` in array dimensions
-------------------------------------------
``f2py`` now supports the Fortran exponentiation operator
``**`` in array dimension specifications such as ``real :: a(2*n, 0:2**n)``.
It previously failed with an ``OpError`` or generated C code that did
not compile.
Fortran wrappers of functions now write the dimensions of their
arguments in Fortran syntax, so C expressions like ``pow(2, n)`` in
signature files no longer break the build.
