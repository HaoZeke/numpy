Empty fixed-form wrappers are program units
-------------------------------------------
``f2py`` created an empty ``.f`` file when a module had no Fortran 77
wrapper. That file is not a program unit, and LFortran stops with an
internal error on it. The placeholder is now a subroutine whose name
is a legal Fortran identifier. Real wrappers still replace the file.
