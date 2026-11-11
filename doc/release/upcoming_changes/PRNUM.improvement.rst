``f2py`` call-backs receive their arguments from undeclaring routines
---------------------------------------------------------------------
A Fortran routine may call an external function that another routine of the same module declares ``intent(callback)``.
Such calls now pass the Fortran arguments to the Python call-back, as calls through the declaring routine do, instead of calling it with only ``<name>_extra_args``.
For backward compatibility, a Python function that can be called with the extra arguments alone, for instance because all of its parameters have default values, is still called with only those.
