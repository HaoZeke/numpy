``f2py`` keeps the procedures of operator and assignment interfaces
-------------------------------------------------------------------
``f2py`` now records every procedure named by the ``module procedure``
statements of a generic interface such as ``interface operator(+)`` or
``interface assignment(=)``. Repeated statements, the optional ``::``
separator and blanks in the generic specification, as in
``operator (+)``, no longer drop or misname procedures. Signature files
written by ``f2py -h`` now contain these ``module procedure``
statements, so the mapping survives a round trip through a ``.pyf``
file.
