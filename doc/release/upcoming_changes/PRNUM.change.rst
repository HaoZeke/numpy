``f2py`` fixes for ``CHARACTER`` arguments and results
------------------------------------------------------
Wrappers now pass the hidden length argument that Fortran compilers expect
for a ``CHARACTER`` dummy argument declared without a length, such as
``character c`` or ``character c(n)``. Without it, a later
``character(len=*)`` argument received a wrong length. As for
``character(len=n)`` arguments, the length is also passed to C functions
wrapped with ``intent(c)``, as a trailing ``size_t`` argument.

Call-back functions now take a ``size_t`` hidden length for every
``CHARACTER`` argument, including ``CHARACTER`` arrays and ``CHARACTER``
declared without a length, in the order the Fortran caller passes them.
Previously a string argument that followed such an argument received the
wrong length. Call-backs returning ``CHARACTER`` and call-backs with
``intent(inout)`` string arguments no longer fail to compile with current GCC
and Clang.

Modules built with ``--no-wrap-functions`` that contain a
``character(len=n)`` function compile again.

``f2py`` reports command line errors and its backend accurately
---------------------------------------------------------------
``f2py`` now exits with status 2 when it is given an unknown option, the same
status ``argparse`` uses for the options it parses, and with status 1 when it
refuses to overwrite an existing signature file. Both used to exit with status
0, so scripts and build tools could not tell that nothing was generated.

``f2py -c`` no longer prints that it cannot use the distutils backend on every
run. ``--backend`` now defaults to ``meson``, and only ``--backend distutils``,
which still falls back to meson, says that the distutils backend was removed.
