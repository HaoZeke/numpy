Assigning to an f2py module parameter raises ``AttributeError``
---------------------------------------------------------------
Assigning to a module parameter through its f2py wrapper now raises ``AttributeError``.
The parameter is a named constant, so the assignment cannot change the Fortran value.
An array parameter is marked read-only, so writing an element raises ``ValueError``.
