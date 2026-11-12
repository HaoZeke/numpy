``f2py`` wraps routines that have internal procedures
-----------------------------------------------------
``f2py`` no longer copies the internal procedures of a subroutine or function (those after ``contains``), or their ``use`` statements, into the Fortran 90 wrapper it writes for assumed-shape array arguments.
Such routines previously failed to compile.
