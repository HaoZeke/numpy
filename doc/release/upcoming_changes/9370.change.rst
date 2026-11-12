``f2py`` character data assignment honours the declared length
---------------------------------------------------------------
Assigning to a ``CHARACTER*(n)`` common block member or module variable of
an ``f2py``-wrapped extension, for example ``mod.block.name = b"abc"``, now
converts the value to the declared length ``n``, the same as an in-place
assignment to the exposed array: longer values are truncated and shorter ones
are padded with null bytes. Previously the length of the assigned value was
used, which left stale trailing characters in the Fortran variable or wrote
past its end into adjacent Fortran data.
