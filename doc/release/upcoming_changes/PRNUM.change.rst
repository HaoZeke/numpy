``f2py`` uses the type written before ``FUNCTION``
--------------------------------------------------
The kind, length or derived type written before ``FUNCTION`` now declares
the ``RESULT`` variable when there is one, as in
``real(kind=8) function f(x) result(r)``. ``f2py`` used the default kind or
length instead, so such a function returned a wrong value outside a
module, lost precision inside a module, and crashed for ``character``
results.

A kind written before ``FUNCTION`` now selects the same C type as in a
type declaration. ``complex(kind=8)`` functions declared this way returned
wrong values, and an ``integer(kind=8)`` function could fail to compile.

``f2py`` reports command line errors and its backend accurately
---------------------------------------------------------------
``f2py`` now exits with status 2 when it is given an unknown option, the same
status ``argparse`` uses for the options it parses, and with status 1 when it
refuses to overwrite an existing signature file. Both used to exit with status
0, so scripts and build tools could not tell that nothing was generated.

``f2py -c`` no longer prints that it cannot use the distutils backend on every
run. ``--backend`` now defaults to ``meson``, and only ``--backend distutils``,
which still falls back to meson, says that the distutils backend was removed.
