``f2py`` uses the signature file when ``-m`` or Fortran sources are given
-------------------------------------------------------------------------
Without ``-c``, ``f2py`` now generates the wrappers from a signature file
(``.pyf``) alone, as ``f2py -c`` already did. ``-m`` and any Fortran sources
passed with the signature file are ignored, and the ignored sources are
reported. Previously ``f2py -m mod mod.pyf mod.f`` also built a module from
``mod.f`` that overwrote ``modmodule.c``, losing ``.pyf`` directives such as
``intent(out)``, and ``f2py -m mod mod.pyf`` wrote an empty module.
