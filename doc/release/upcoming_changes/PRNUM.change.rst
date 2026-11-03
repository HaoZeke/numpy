``f2py`` maps ``c_size_t``, ``c_intptr_t`` and ``c_ptrdiff_t`` to ``intp``
--------------------------------------------------------------------------
Fortran integers of these ``iso_c_binding`` kinds are now converted to and
from `numpy.intp` by ``f2py`` wrappers.
Previously ``c_size_t`` was treated as a 32-bit unsigned integer, so arrays of
that kind were too small for the data the Fortran code read and wrote, and
scalar arguments of that kind did not compile.
``c_intptr_t`` and ``c_ptrdiff_t`` were treated as C ``long``, which is 32 bits
on Windows.
