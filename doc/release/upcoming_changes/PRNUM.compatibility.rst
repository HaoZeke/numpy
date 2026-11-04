External array-valued functions use explicit interfaces
------------------------------------------------------------
F2PY generates an explicit Fortran interface when wrapping an external
function with an array result, including functions with fixed-shape
arguments. This preserves the result's array rank at the generated call
site. In particular, a CHARACTER array result is not declared as a scalar
CHARACTER result whose length is the array extent.
