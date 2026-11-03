``f2py`` no longer crashes on Fortran objects with nothing to wrap
------------------------------------------------------------------
Reading ``_cpointer`` from an ``f2py`` object without data, such as an
unallocated allocatable array that is the only variable of a Fortran 90
module, or a routine declared with an empty ``fortranname``, now raises
``AttributeError`` instead of crashing.
