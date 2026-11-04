``f2py`` writes back scalar ``intent(inout)`` arguments with NumPy casting
--------------------------------------------------------------------------
A scalar ``intent(inout)`` argument passed as an array is now updated through NumPy's array assignment with unsafe casting.
Arrays in non-native byte order no longer receive garbage, and ``float16`` and other dtypes that were silently left unchanged are updated.
Complex results stored in boolean arrays follow the usual truth value, and object arrays no longer leak a reference on every call.
A read-only array now raises ``ValueError`` instead of being modified.
``integer(1)`` arguments with ``intent(inout)`` also compile now.

External array-valued functions use explicit interfaces
------------------------------------------------------------
F2PY generates an explicit Fortran interface when wrapping an external
function with an array result, including functions with fixed-shape
arguments. This preserves the result's array rank at the generated call
site. In particular, a CHARACTER array result is not declared as a scalar
CHARACTER result whose length is the array extent.
