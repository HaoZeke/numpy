``f2py`` keeps C operator precedence in dimension specifications
-----------------------------------------------------------------
C logical operators (``&&``, ``||`` and ``!``) and casts such as
``(int)v`` in the ``dimension`` specifications of signature files are now
parsed with their C precedence and no longer emit ``ExprWarning``.
Previously they could be reordered with arithmetic and comparisons, so
that, for example, ``*range=='I' && iu-il+1==n`` was compiled as
``*range == 1 + 'I' && iu - il == n`` and produced a wrongly sized array,
and ``(int)(x+y)`` was compiled as the invalid C ``int(x + y)``.

``f2py`` reports command line errors and its backend accurately
---------------------------------------------------------------
``f2py`` now exits with status 2 when it is given an unknown option, the same
status ``argparse`` uses for the options it parses, and with status 1 when it
refuses to overwrite an existing signature file. Both used to exit with status
0, so scripts and build tools could not tell that nothing was generated.

``f2py -c`` no longer prints that it cannot use the distutils backend on every
run. ``--backend`` now defaults to ``meson``, and only ``--backend distutils``,
which still falls back to meson, says that the distutils backend was removed.
