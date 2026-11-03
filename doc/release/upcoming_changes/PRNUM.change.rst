``f2py -c`` passes ``-D`` and ``-U`` macros to the compilers again
------------------------------------------------------------------
The meson backend of ``f2py -c`` silently ignored ``-D<macro>[=<defn>]``
and ``-U<macro>``. They are now passed to the C and Fortran compilers,
as with the former ``distutils`` backend, so macros such as
``-DF2PY_REPORT_ATEXIT`` or ``-DUPPERCASE_FORTRAN`` and ``#ifdef`` guards
in preprocessed Fortran sources take effect again.

An exception raised in a callback no longer aborts Python when the module
is built with ``-DF2PY_USE_PYTHON_TLS``, or when ``f2py`` finds no
thread-local storage support in the C compiler.
