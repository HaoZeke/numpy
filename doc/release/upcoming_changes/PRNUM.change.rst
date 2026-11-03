``f2py`` fixes for ``CHARACTER`` arguments and results
------------------------------------------------------
Wrappers now pass the hidden length argument that Fortran compilers expect
for a ``CHARACTER`` dummy argument declared without a length, such as
``character c`` or ``character c(n)``. Without it, a later
``character(len=*)`` argument received a wrong length. As for
``character(len=n)`` arguments, the length is also passed to C functions
wrapped with ``intent(c)``, as a trailing ``size_t`` argument.
