F2PY wrappers with ``intent(inout)`` one-byte integer scalars compile
---------------------------------------------------------------------

F2PY now defines the helper ``pyobj_from_signed_char1``.
Generated C code calls that helper for an ``intent(inout)`` scalar.
The scalar is a one-byte integer kind.
The kinds are ``integer*1``, ``c_signed_char``, ``c_int8_t``, ``c_int_least8_t`` and ``int8``.
The helper writes the scalar back into an object array.
Those wrappers used to fail while compiling.
Some builds imported a module whose link was missing the symbol.
