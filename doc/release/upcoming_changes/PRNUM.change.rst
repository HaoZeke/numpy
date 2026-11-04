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

``f2py`` reports command line errors and its backend accurately
---------------------------------------------------------------
``f2py`` now exits with status 2 when it is given an unknown option, the same
status ``argparse`` uses for the options it parses, and with status 1 when it
refuses to overwrite an existing signature file. Both used to exit with status
0, so scripts and build tools could not tell that nothing was generated.

``f2py -c`` no longer prints that it cannot use the distutils backend on every
run. ``--backend`` now defaults to ``meson``, and only ``--backend distutils``,
which still falls back to meson, says that the distutils backend was removed.
