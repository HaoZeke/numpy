``f2py`` wraps call-back arguments that the routine never calls
---------------------------------------------------------------
A routine with an ``external`` argument that it never calls, for example one
that it only passes on to another routine, no longer produces C code that fails
to compile. ``f2py`` gives such a call-back no arguments, making it a function
of its declared type or a subroutine if it has no type, and ``f2py -h`` writes
this signature to the signature file, where its arguments can be added.
