F2PY documentation of wrapper argument order
--------------------------------------------
The F2PY guide now explains how the Python signature of a wrapper function is
derived from the Fortran argument list. Dimension arguments of input arrays
become optional and move behind the required arguments, several
``intent(out)`` arguments are returned as a tuple, and the ``required``
attribute keeps a dimension argument in its Fortran position.
