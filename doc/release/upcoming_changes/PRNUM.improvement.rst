``f2py`` wraps implicitly typed ``ENTRY`` points
------------------------------------------------
``f2py`` no longer raises ``KeyError`` for a Fortran function with an
``ENTRY`` statement whose name has no type declaration.
The entry is wrapped with the type given by the implicit typing rules.
