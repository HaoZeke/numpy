F2PY uses the C types of more ``iso_c_binding`` kinds
-----------------------------------------------------

F2PY now declares these ``iso_c_binding`` kinds with the C types they interoperate with.
The kinds are ``c_size_t``, ``c_ptrdiff_t``, ``c_intptr_t``, ``c_intmax_t``, ``c_int_fast8_t``, ``c_int_fast16_t``, ``c_int_fast32_t``, ``c_int_fast64_t`` and ``c_bool``.
The C types are ``size_t``, ``ptrdiff_t``, ``intptr_t``, ``intmax_t``, ``int_fast8_t``, ``int_fast16_t``, ``int_fast32_t``, ``int_fast64_t`` and ``bool``.
The C implementation chooses the width of these types.
Fixed-width aliases were wrong on some platforms.
Wrappers for ``c_size_t`` and ``c_bool`` failed to compile.
``c_int_fast16_t`` and ``c_int_fast32_t`` were passed as 2-byte and 4-byte C values.
On 64-bit glibc those kinds are 8 bytes wide.
``c_intptr_t`` and ``c_ptrdiff_t`` used ``long``.
On 64-bit Windows that ``long`` is 4 bytes.

An array of one of these kinds now uses the NumPy integer dtype of the C type's width.
``c_size_t`` is ``uint64`` on a 64-bit platform.
An array of ``logical(c_bool)`` has dtype ``bool``.
``.f2py_f2cmap`` files accept these C type names.

A scalar ``c_size_t`` input goes through ``long long``.
A value above ``LLONG_MAX`` is rejected.
A scalar ``logical(c_bool)`` result is returned as a Python ``int``.

LLVM Flang gives ``c_int_fast16_t`` and ``c_int_fast32_t`` a width that differs from the C types ``int_fast16_t`` and ``int_fast32_t``.
On Linux, Flang 19 reports 2 bytes and 4 bytes for those two kinds.
The open LLVM issue records the same widths on Windows.
The issue is ``[Flang] Incorrect value of C_INT_FAST*_T`` (`llvm/llvm-project#208361 <https://github.com/llvm/llvm-project/issues/208361>`__).
With that compiler, call ``c_int16_t`` and ``c_int32_t`` instead.

External array-valued functions use explicit interfaces
------------------------------------------------------------
F2PY generates an explicit Fortran interface when wrapping an external
function with an array result, including functions with fixed-shape
arguments. This preserves the result's array rank at the generated call
site. In particular, a CHARACTER array result is not declared as a scalar
CHARACTER result whose length is the array extent.
