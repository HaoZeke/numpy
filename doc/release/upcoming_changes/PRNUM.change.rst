``f2py`` keeps names that follow a backslash in a string literal
----------------------------------------------------------------
``f2py`` no longer treats a backslash as an escape while it masks Fortran string literals during dependency scans.
A quote is escaped by doubling it, so an initializer such as ``'C:\'//name`` still depends on ``name``.
