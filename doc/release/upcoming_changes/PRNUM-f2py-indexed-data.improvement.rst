Element-indexed Fortran DATA in modules
---------------------------------------
F2PY associates element-indexed DATA statements with their declared array.
Separate and combined constant-index initializers generate a valid signature
and preserve initialized module values, including repeated DATA values.
