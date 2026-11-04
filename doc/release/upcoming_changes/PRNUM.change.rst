``f2py`` uses the signature file when ``-m`` or Fortran sources are given
-------------------------------------------------------------------------
Without ``-c``, ``f2py`` now generates the wrappers from a signature file
(``.pyf``) alone, as ``f2py -c`` already did. ``-m`` and any Fortran sources
passed with the signature file are ignored, and the ignored sources are
reported. Previously ``f2py -m mod mod.pyf mod.f`` also built a module from
``mod.f`` that overwrote ``modmodule.c``, losing ``.pyf`` directives such as
``intent(out)``, and ``f2py -m mod mod.pyf`` wrote an empty module.

``f2py`` reports command line errors and its backend accurately
---------------------------------------------------------------
``f2py`` now exits with status 2 when it is given an unknown option, the same
status ``argparse`` uses for the options it parses, and with status 1 when it
refuses to overwrite an existing signature file. Both used to exit with status
0, so scripts and build tools could not tell that nothing was generated.

``f2py -c`` no longer prints that it cannot use the distutils backend on every
run. ``--backend`` now defaults to ``meson``, and only ``--backend distutils``,
which still falls back to meson, says that the distutils backend was removed.
