``f2py`` tests no longer index empty Fortran arrays
---------------------------------------------------
The Fortran helpers of ``TestDimSpec`` in
``numpy/f2py/tests/test_crackfortran.py`` no longer reference an element
of a zero-size array. The out-of-bounds reference was reported by valgrind
as a use of uninitialised memory, and it aborted the test run when the
tests were compiled with ``gfortran -fcheck=bounds``.

``f2py`` reports command line errors and its backend accurately
---------------------------------------------------------------
``f2py`` now exits with status 2 when it is given an unknown option, the same
status ``argparse`` uses for the options it parses, and with status 1 when it
refuses to overwrite an existing signature file. Both used to exit with status
0, so scripts and build tools could not tell that nothing was generated.

``f2py -c`` no longer prints that it cannot use the distutils backend on every
run. ``--backend`` now defaults to ``meson``, and only ``--backend distutils``,
which still falls back to meson, says that the distutils backend was removed.
