``f2py`` no longer prints debug output for derived types
--------------------------------------------------------
Parsing a derived type definition no longer prints debug lines to
standard output, which corrupted the signature file written by
``f2py -h stdout``.
Generated signature files also no longer declare the names of derived
types as implicitly typed variables.
