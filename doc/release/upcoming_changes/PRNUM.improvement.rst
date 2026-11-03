``f2py`` call-backs shared between routines
-------------------------------------------
Several routines of a module can now declare the same external ``intent(callback)``.
They share one call-back; previously the generated C code defined it once per routine and failed to compile.

A routine that calls such an external without declaring it now passes the Fortran arguments to the Python call-back, as the declaring routines do, instead of calling it with only ``<name>_extra_args``.
For backward compatibility, a Python function that can be called with the extra arguments alone, for instance because all of its parameters have default values, is still called with only those from such a routine.
