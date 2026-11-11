``f2py`` keeps ``IMPLICIT`` statements in their program unit
------------------------------------------------------------
An ``IMPLICIT`` statement in one Fortran program unit no longer changes
the types that ``f2py`` gives to implicitly typed variables of the
program units read after it.
