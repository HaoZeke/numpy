``f2py`` passes all arguments to call-backs that take ``*args``
---------------------------------------------------------------
A Python call-back with a ``*args`` parameter now receives every Fortran argument.
An example is ``def fun(a, *args)``.
Those arguments are followed by ``<name>_extra_args``.
Previously only the named parameters were counted.
Fortran arguments were silently dropped or replaced by the extra arguments.
A bound method taking only ``*args`` crashed the interpreter.
