``f2py`` rejects ``REAL(KIND=16)`` where C ``long double`` differs
--------------------------------------------------------------------
``f2py`` passes ``REAL(KIND=16)`` and ``COMPLEX(KIND=16)`` arguments as
C ``long double``, which has the same format only where it is IEEE
binary128 or IBM double-double.
Elsewhere, for example on x86-64, calling such a wrapper now raises
``NotImplementedError`` instead of crashing or returning reinterpreted
values.
``REAL(KIND=C_LONG_DOUBLE)`` is the portable extended precision kind.
