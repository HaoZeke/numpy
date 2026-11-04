``f2py`` converts ``str_`` arrays passed as ``character`` arguments
-------------------------------------------------------------------
A `numpy.str_` array passed to a scalar ``character`` or
``character(len=n)`` argument of an ``f2py``-wrapped routine is now
converted with NumPy's ASCII cast, as for character array arguments.
Previously ``f2py`` read the UCS4 buffer as bytes, so little-endian
systems passed only the first character, big-endian systems passed an
empty string, and ``intent(inout)`` results corrupted the array.
Non-ASCII characters now raise an error, as for `str` arguments.

``f2py`` reports command line errors and its backend accurately
---------------------------------------------------------------
``f2py`` now exits with status 2 when it is given an unknown option, the same
status ``argparse`` uses for the options it parses, and with status 1 when it
refuses to overwrite an existing signature file. Both used to exit with status
0, so scripts and build tools could not tell that nothing was generated.

``f2py -c`` no longer prints that it cannot use the distutils backend on every
run. ``--backend`` now defaults to ``meson``, and only ``--backend distutils``,
which still falls back to meson, says that the distutils backend was removed.
