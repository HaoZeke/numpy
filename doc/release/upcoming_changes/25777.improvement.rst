``f2py`` module wrappers no longer import unused allocatable arrays
-------------------------------------------------------------------
The ``f2pyinit<module>`` routine that ``f2py`` writes for a Fortran 90 module no longer use-associates the module's allocatable arrays, which it only reaches through their accessor routines.
Compiling the generated wrapper with ``gfortran -Wall`` therefore no longer warns about an unused module variable for every allocatable array.
