``f2py`` passes all arguments to call-backs that take ``*args``
---------------------------------------------------------------
A Python call-back with a ``*args`` parameter, such as ``def fun(a, *args)``,
now receives every argument from the Fortran routine followed by the
``<name>_extra_args``. Previously only the named parameters were counted, so
arguments from Fortran were silently dropped or replaced by the extra
arguments, and a bound method taking only ``*args`` crashed the interpreter.

``f2py`` reports command line errors and its backend accurately
---------------------------------------------------------------
``f2py`` now exits with status 2 when it is given an unknown option, the same
status ``argparse`` uses for the options it parses, and with status 1 when it
refuses to overwrite an existing signature file. Both used to exit with status
0, so scripts and build tools could not tell that nothing was generated.

``f2py -c`` no longer prints that it cannot use the distutils backend on every
run. ``--backend`` now defaults to ``meson``, and only ``--backend distutils``,
which still falls back to meson, says that the distutils backend was removed.
