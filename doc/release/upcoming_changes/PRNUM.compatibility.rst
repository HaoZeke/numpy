f2py scalar conversions preserve numeric protocol errors
--------------------------------------------------------
For integer and real scalar arguments, f2py propagates errors from an
object's numeric conversion protocol, including warnings raised as errors.
For example, a failed ``__float__``, ``__int__`` or ``__index__`` call does
not cause f2py to retry conversion using the object's first element.
This also preserves errors raised by NumPy array scalar conversions.

The supported first-element conversion for sequences without a relevant
numeric conversion protocol, and real-part conversion for built-in complex
objects, remain available. These conversions emit no additional warnings.

External array-valued functions use explicit interfaces
------------------------------------------------------------
F2PY generates an explicit Fortran interface when wrapping an external
function with an array result, including functions with fixed-shape
arguments. This preserves the result's array rank at the generated call
site. In particular, a CHARACTER array result is not declared as a scalar
CHARACTER result whose length is the array extent.
