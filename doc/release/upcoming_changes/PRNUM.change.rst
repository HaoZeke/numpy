``f2py`` evaluates ``PARAMETER`` values that contain real literals
------------------------------------------------------------------
Since NumPy 1.23, a named constant whose value contained a real literal was replaced by a kind number (4 or 8) before it was evaluated, so ``integer, parameter :: n = int(1.0e2)`` was taken as ``n = 4``.
When such a constant sized an argument, the wrapper allocated or checked the wrong number of elements and the Fortran routine could write past the end of the array.
These values are now evaluated first; only expressions that ``f2py`` cannot evaluate, such as ``selected_real_kind(precision(1.0d0))``, still fall back to the kind of their real literal.

``f2py`` reports command line errors and its backend accurately
---------------------------------------------------------------
``f2py`` now exits with status 2 when it is given an unknown option, the same
status ``argparse`` uses for the options it parses, and with status 1 when it
refuses to overwrite an existing signature file. Both used to exit with status
0, so scripts and build tools could not tell that nothing was generated.

``f2py -c`` no longer prints that it cannot use the distutils backend on every
run. ``--backend`` now defaults to ``meson``, and only ``--backend distutils``,
which still falls back to meson, says that the distutils backend was removed.
