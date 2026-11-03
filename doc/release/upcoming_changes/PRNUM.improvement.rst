``f2py`` derives ``callprotoargument`` from ``callstatement``
-------------------------------------------------------------
Signature files that use ``callstatement`` without ``callprotoargument`` now
produce wrappers that compile as C23, the default of GCC 15. ``f2py`` takes the
prototype of the wrapped routine from the arguments passed to
``(*f2py_func)`` when each one is a variable of the signature, its address, or
a call-back function, instead of declaring an empty parameter list, which C23
reads as ``(void)``.
