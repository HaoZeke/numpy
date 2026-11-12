``f2py`` explains tuples returned by call-back functions
--------------------------------------------------------
``f2py`` unpacks a tuple returned by a Python call-back function into
the call-back's return values.
Returning ``(a, b, c)`` from a call-back whose only return value is an
array therefore converted ``a`` alone and failed with an unrelated
dimension error.
If a call-back returns a tuple with more items than it has return values
and a return value then fails to convert, ``f2py`` now raises an error
that names the expected call-back signature and suggests returning a
list or an array.
The original conversion error is kept as its ``__cause__``.
