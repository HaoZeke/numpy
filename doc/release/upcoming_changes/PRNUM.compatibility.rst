``f2py`` now stops when a Fortran kind has no NumPy type with the same storage.
``integer(16)``, ``logical(16)``, ``real(3)`` and ``complex(2)`` are kinds that stop the build.
``real(16)`` and ``complex(16)`` map to ``long double`` where that type is IEEE binary128 or IBM double-double.
That covers aarch64 Linux and ppc64le.
On x86-64 the build stops, because ``long double`` is the x87 type and NumPy has no binary128 dtype.
A signature that means C ``long double`` should say ``real(kind=c_long_double)`` rather than ``real*16``.
``real*12`` and ``complex*24`` map to ``long double`` only where that type is 12 bytes wide.
``real(2)`` maps to NumPy ``float16``, and its C type is ``_Float16``.
``selected_real_kind`` knows ``real(2)`` and ``real(3)`` when the compiler in ``FC`` has them.
Kind names such as ``real128`` and ``c_float128`` map as their kind numbers.
``character(kind=4)`` stops the build, because ``f2py`` passes characters as bytes.
``c_size_t`` has the width of ``size_t``, and scalar arguments of unsigned kinds convert.
``c_int_fast16_t`` and ``c_int_fast32_t`` take the kind the Fortran compiler gives them.
A module procedure sees the named constants of its module.
``check()`` prints a ``real(2)`` scalar as a number.

External array-valued functions use explicit interfaces
------------------------------------------------------------
F2PY generates an explicit Fortran interface when wrapping an external
function with an array result, including functions with fixed-shape
arguments. This preserves the result's array rank at the generated call
site. In particular, a CHARACTER array result is not declared as a scalar
CHARACTER result whose length is the array extent.
