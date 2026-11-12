``f2py`` builds with LFortran's separate-compilation mode
---------------------------------------------------------
LFortran does not use gfortran's defaults. ``f2py`` now passes
``--implicit-typing``, ``--implicit-interface``,
``--mangle-underscore-external``, ``--fixed-form-infer``,
``--separate-compilation``, and ``--print-leading-space`` when the
Fortran compiler is LFortran, so external symbols, module procedures,
fixed-form ``.f`` files, and list-directed output match the other
compilers.
