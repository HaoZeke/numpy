``f2py`` wrapper interfaces match the wrapped Fortran routine
-------------------------------------------------------------
A Fortran function without a ``result`` clause that takes an
assumed-shape argument now returns the correct value.
The Fortran wrapper that ``f2py`` generates for it declared no type for
the function, so the result was read with an implicit type.
