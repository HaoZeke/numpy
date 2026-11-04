``f2py -c`` links prebuilt libraries after the wrapper objects
--------------------------------------------------------------
Static (``.a``) and shared (``.so``, ``.dylib``) libraries passed to
``f2py -c`` on the command line are now linked after the generated wrapper
objects, from where they are. Previously the meson backend copied them into
its temporary build directory and placed them before the wrapper objects, so
their symbols were missing when the module was imported. Object files given
with a directory are now found as well.

``f2py`` reports command line errors and its backend accurately
---------------------------------------------------------------
``f2py`` now exits with status 2 when it is given an unknown option, the same
status ``argparse`` uses for the options it parses, and with status 1 when it
refuses to overwrite an existing signature file. Both used to exit with status
0, so scripts and build tools could not tell that nothing was generated.

``f2py -c`` no longer prints that it cannot use the distutils backend on every
run. ``--backend`` now defaults to ``meson``, and only ``--backend distutils``,
which still falls back to meson, says that the distutils backend was removed.
