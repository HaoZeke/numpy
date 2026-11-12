``f2py`` applies each ``IMPLICIT`` statement to its own program unit
--------------------------------------------------------------------
An ``IMPLICIT`` statement in one Fortran program unit no longer changes
the types that ``f2py`` gives to implicitly typed variables of the
program units read after it.
``IMPLICIT`` letters are also matched regardless of case, so an
uppercase ``IMPLICIT`` statement is no longer ignored when ``f2py`` keeps
the case of the source, as it does for ``f2py -m`` without ``-h``.
