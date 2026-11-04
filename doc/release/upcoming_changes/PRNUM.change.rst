``f2py -c`` modules can call shared-library procedures with Intel Fortran
--------------------------------------------------------------------------
Extension modules built with ``f2py -c`` are compiled with default symbol
visibility again, as they were with the removed ``distutils`` backend. Meson
hides the symbols of Python extension modules by default, and Intel's ``ifx``
applies ``-fvisibility=hidden`` to external references as well as definitions,
so Fortran code that called procedures from shared libraries through explicit
interfaces, such as the OpenMP routines in ``omp_lib``, failed to link with
``hidden symbol ... isn't defined``.

``f2py`` reports command line errors and its backend accurately
---------------------------------------------------------------
``f2py`` now exits with status 2 when it is given an unknown option, the same
status ``argparse`` uses for the options it parses, and with status 1 when it
refuses to overwrite an existing signature file. Both used to exit with status
0, so scripts and build tools could not tell that nothing was generated.

``f2py -c`` no longer prints that it cannot use the distutils backend on every
run. ``--backend`` now defaults to ``meson``, and only ``--backend distutils``,
which still falls back to meson, says that the distutils backend was removed.
