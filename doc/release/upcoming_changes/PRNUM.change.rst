``f2py`` module function wrappers keep all their arguments
----------------------------------------------------------
The wrapper of a Fortran 90 module function no longer drops arguments whose
names end with the function name, which made such functions return wrong
results.
