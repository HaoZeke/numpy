Fortran compiler wrappers supply extension link options
-------------------------------------------------------
The Meson backend of ``f2py`` uses its selected Fortran compiler to link
generated extensions.
Compiler wrappers selected through ``FC``, including MPI wrappers, can
supply their libraries at link time as well as their module files and
include directories at compile time.
