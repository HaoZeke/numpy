``f2py`` wraps ``logical(c_bool)`` variables
--------------------------------------------
Fortran routines with ``logical(c_bool)`` scalar or array arguments from
``iso_c_binding`` can now be wrapped; the generated C code used to refer to an
undeclared ``unsigned_char`` type and failed to compile. ``logical(c_bool)`` is
now handled like ``logical(kind=1)``.
