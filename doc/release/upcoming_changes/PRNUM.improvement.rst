``f2py`` no longer crashes on Fortran objects with nothing to wrap
------------------------------------------------------------------
Reading ``_cpointer`` from an ``f2py`` object without data, such as an
unallocated allocatable array that is the only variable of a Fortran 90
module, or a routine declared with an empty ``fortranname``, now raises
``AttributeError`` instead of crashing.
Fortran 90 modules that only define derived types or interfaces are now
skipped, like modules without public entities, instead of crashing on
import.
