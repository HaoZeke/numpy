``f2py`` uses the type written before ``FUNCTION``
--------------------------------------------------
The kind, length or derived type written before ``FUNCTION`` now declares
the ``RESULT`` variable when there is one, as in
``real(kind=8) function f(x) result(r)``. ``f2py`` used the default kind or
length instead, so such a function returned a wrong value outside a
module, lost precision inside a module, and crashed for ``character``
results.

A kind written before ``FUNCTION`` now selects the same C type as in a
type declaration. ``complex(kind=8)`` functions declared this way returned
wrong values, and an ``integer(kind=8)`` function could fail to compile.
