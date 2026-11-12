``f2py`` exports ``intent(callback)`` functions from the module
---------------------------------------------------------------
The C functions that ``f2py`` generates for ``intent(callback)`` externals are now exported from the extension module, also when it is built with hidden symbol visibility as meson does by default.
Fortran code in a separate shared library that calls such a function by name can now call back into Python.
