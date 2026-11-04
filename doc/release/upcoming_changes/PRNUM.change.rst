``f2py`` maps ``c_size_t``, ``c_intptr_t`` and ``c_ptrdiff_t`` to ``intp``
--------------------------------------------------------------------------
Fortran integers of these ``iso_c_binding`` kinds are now converted to and
from `numpy.intp` by ``f2py`` wrappers.
Previously ``c_size_t`` was treated as a 32-bit unsigned integer, so arrays of
that kind were too small for the data the Fortran code read and wrote, and
scalar arguments of that kind did not compile.
``c_intptr_t`` and ``c_ptrdiff_t`` were treated as C ``long``, which is 32 bits
on Windows.

``f2py`` reports command line errors and its backend accurately
---------------------------------------------------------------
``f2py`` now exits with status 2 when it is given an unknown option, the same
status ``argparse`` uses for the options it parses, and with status 1 when it
refuses to overwrite an existing signature file. Both used to exit with status
0, so scripts and build tools could not tell that nothing was generated.

``f2py -c`` no longer prints that it cannot use the distutils backend on every
run. ``--backend`` now defaults to ``meson``, and only ``--backend distutils``,
which still falls back to meson, says that the distutils backend was removed.
