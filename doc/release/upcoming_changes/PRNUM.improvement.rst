``f2py`` wraps call-back arguments that the routine never calls
---------------------------------------------------------------
An ``external`` argument that a routine never calls now gets a wrapper.
The usual case is an argument the routine only passes on.
Generated C for that call-back no longer fails to compile.
``f2py`` gives such a call-back no arguments.
A typed call-back becomes a function of its declared type.
An untyped call-back becomes a subroutine.
``f2py -h`` writes this signature to the signature file.
The arguments can be added there.
