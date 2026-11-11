``f2py`` wrapper interfaces match the wrapped Fortran routine
-------------------------------------------------------------
A Fortran function without a ``result`` clause that takes an
assumed-shape argument now returns the correct value.
The Fortran wrapper that ``f2py`` generates for it declared no type for
the function, so the result was read with an implicit type.

Building a routine with assumed-shape arguments from a signature file,
such as one written by ``f2py -h``, no longer fails to compile.
The Fortran wrapper no longer contains ``f2py``-only syntax such as
``check``, ``depend``, ``required``, ``intent(hide)``, argument default
values, ``threadsafe`` or ``fortranname``, which also came from
``!f2py`` directives in Fortran sources.
