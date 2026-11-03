``f2py`` uses the type written before ``FUNCTION``
--------------------------------------------------
A kind written before ``FUNCTION`` now selects the same C type as in a
type declaration. ``complex(kind=8)`` functions declared this way returned
wrong values, and an ``integer(kind=8)`` function could fail to compile.
