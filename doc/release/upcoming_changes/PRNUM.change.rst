``f2py`` wraps assumed-shape arrays that declare a lower bound
--------------------------------------------------------------
Assumed-shape dummy arguments with an explicit lower bound, such as
``a(0:)``, no longer make ``f2py`` generate C code that fails to compile.
They are now wrapped like ``a(:)``, and the declared lower bound still
applies inside the Fortran procedure.

``f2py`` reports command line errors and its backend accurately
---------------------------------------------------------------
``f2py`` now exits with status 2 when it is given an unknown option, the same
status ``argparse`` uses for the options it parses, and with status 1 when it
refuses to overwrite an existing signature file. Both used to exit with status
0, so scripts and build tools could not tell that nothing was generated.

``f2py -c`` no longer prints that it cannot use the distutils backend on every
run. ``--backend`` now defaults to ``meson``, and only ``--backend distutils``,
which still falls back to meson, says that the distutils backend was removed.
