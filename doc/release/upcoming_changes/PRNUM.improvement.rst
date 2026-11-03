``f2py`` returns ``long double`` scalars instead of crashing
------------------------------------------------------------
Wrapped routines whose scalar results are C ``long double``, such as
``real(kind=c_long_double), intent(out)`` arguments and functions, now return a
Python float. Previously the value reached ``Py_BuildValue`` as an object
pointer, which raised ``SystemError`` or crashed the interpreter.
