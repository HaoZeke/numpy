``f2py`` decodes ``str`` arrays passed to ``character`` arguments
-----------------------------------------------------------------
A NumPy array of dtype ``U`` passed to a scalar ``character`` or ``character(len=n)`` argument of an ``f2py`` wrapper is now read in the byte order of its dtype and encoded as ASCII, as a ``str`` argument is.
Previously its UCS4 data was copied as raw bytes: on big-endian platforms the argument arrived blank, and on little-endian platforms only its first character was passed.
Arrays holding non-ASCII characters are now rejected, like ``str`` arguments.
For ``intent(inout)`` arguments, the result is now written back to such arrays as characters rather than as raw bytes.

``f2py`` reports command line errors and its backend accurately
---------------------------------------------------------------
``f2py`` now exits with status 2 when it is given an unknown option, the same
status ``argparse`` uses for the options it parses, and with status 1 when it
refuses to overwrite an existing signature file. Both used to exit with status
0, so scripts and build tools could not tell that nothing was generated.

``f2py -c`` no longer prints that it cannot use the distutils backend on every
run. ``--backend`` now defaults to ``meson``, and only ``--backend distutils``,
which still falls back to meson, says that the distutils backend was removed.
