``f2py`` documents call-backs from separately built Fortran code
----------------------------------------------------------------
The ``f2py`` documentation now shows how Fortran code that is built
separately, for example into a static library, calls Python functions
through a wrapped entry point, both for call-backs passed as procedure
arguments and for call-backs with a fixed external name.
