``f2py`` reuses a call-back signature when an ``external`` is passed on
-----------------------------------------------------------------------
A routine that does not call an ``external`` argument, but passes it to
another routine in the same source, now gives that argument the callee's
signature. The generated signature file renames it with ``use``, for example
``use fun1__user__routines, myfun2=>myfun1``. A call-back with no call and no
such callee still has no arguments. An interface block or an edited signature
file is left as written.
