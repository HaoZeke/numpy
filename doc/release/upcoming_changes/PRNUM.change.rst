``f2py`` passes all arguments to call-backs that take ``*args``
---------------------------------------------------------------
A Python call-back with a ``*args`` parameter, such as ``def fun(a, *args)``,
now receives every argument from the Fortran routine followed by the
``<name>_extra_args``. Previously only the named parameters were counted, so
arguments from Fortran were silently dropped or replaced by the extra
arguments, and a bound method taking only ``*args`` crashed the interpreter.
