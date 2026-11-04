``f2py`` applies each ``IMPLICIT`` statement to its own program unit
--------------------------------------------------------------------
An ``IMPLICIT`` statement in one Fortran program unit no longer changes
the types that ``f2py`` gives to implicitly typed variables of the
program units read after it.
``IMPLICIT`` letters are also matched regardless of case, so an
uppercase ``IMPLICIT`` statement is no longer ignored when ``f2py`` keeps
the case of the source, as it does for ``f2py -m`` without ``-h``.

``f2py`` reports command line errors and its backend accurately
---------------------------------------------------------------
``f2py`` now exits with status 2 when it is given an unknown option, the same
status ``argparse`` uses for the options it parses, and with status 1 when it
refuses to overwrite an existing signature file. Both used to exit with status
0, so scripts and build tools could not tell that nothing was generated.

``f2py -c`` no longer prints that it cannot use the distutils backend on every
run. ``--backend`` now defaults to ``meson``, and only ``--backend distutils``,
which still falls back to meson, says that the distutils backend was removed.
