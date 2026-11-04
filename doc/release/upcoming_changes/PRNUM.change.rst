``f2py`` character data assignment honours the declared length
---------------------------------------------------------------
Assigning to a ``CHARACTER*(n)`` common block member or module variable of
an ``f2py``-wrapped extension, for example ``mod.block.name = b"abc"``, now
converts the value to the declared length ``n``, the same as an in-place
assignment to the exposed array: longer values are truncated and shorter ones
are padded with null bytes. Previously the length of the assigned value was
used, which left stale trailing characters in the Fortran variable or wrote
past its end into adjacent Fortran data.

``f2py`` reports command line errors and its backend accurately
---------------------------------------------------------------
``f2py`` now exits with status 2 when it is given an unknown option, the same
status ``argparse`` uses for the options it parses, and with status 1 when it
refuses to overwrite an existing signature file. Both used to exit with status
0, so scripts and build tools could not tell that nothing was generated.

``f2py -c`` no longer prints that it cannot use the distutils backend on every
run. ``--backend`` now defaults to ``meson``, and only ``--backend distutils``,
which still falls back to meson, says that the distutils backend was removed.
