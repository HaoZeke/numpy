``f2py`` wraps assumed-shape arrays that declare a lower bound
--------------------------------------------------------------
Assumed-shape dummy arguments with an explicit lower bound, such as ``a(0:)``, no longer make ``f2py`` generate C code that fails to compile.
They are now wrapped like ``a(:)``, and the declared lower bound still applies inside the Fortran procedure.
