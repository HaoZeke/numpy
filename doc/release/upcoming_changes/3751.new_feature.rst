``f2py --strict-setattr`` rejects assignments that Fortran cannot see
---------------------------------------------------------------------
``f2py`` has a new ``--[no-]strict-setattr`` option.
With ``--strict-setattr``, the objects that wrap Fortran 90 modules and COMMON blocks raise ``AttributeError`` on assignment to a name that is not a wrapped Fortran variable.
A misspelled variable, or a variable missing from the signature file, is such a name.
The default, ``--no-strict-setattr``, keeps the previous behavior of storing such a value as a Python attribute that the Fortran code never sees.
