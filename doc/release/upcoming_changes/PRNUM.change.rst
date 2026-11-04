``f2py`` rejects ``REAL(KIND=16)`` where C ``long double`` differs
--------------------------------------------------------------------
``f2py`` passes ``REAL(KIND=16)`` and ``COMPLEX(KIND=16)`` arguments as C
``long double``, which has the same format only where it is IEEE binary128 or
IBM double-double. Elsewhere, for example on x86-64, calling such a wrapper now
raises ``NotImplementedError`` instead of crashing or returning reinterpreted
values. ``REAL(KIND=C_LONG_DOUBLE)`` is the portable extended precision kind.

``f2py`` reports command line errors and its backend accurately
---------------------------------------------------------------
``f2py`` now exits with status 2 when it is given an unknown option, the same
status ``argparse`` uses for the options it parses, and with status 1 when it
refuses to overwrite an existing signature file. Both used to exit with status
0, so scripts and build tools could not tell that nothing was generated.

``f2py -c`` no longer prints that it cannot use the distutils backend on every
run. ``--backend`` now defaults to ``meson``, and only ``--backend distutils``,
which still falls back to meson, says that the distutils backend was removed.
