F2PY wrappers with ``intent(inout)`` one-byte integer scalars compile
---------------------------------------------------------------------
F2PY now defines the ``pyobj_from_signed_char1`` helper that the generated
C code uses to write an ``intent(inout)`` scalar of a one-byte integer kind
(``integer*1``, ``c_signed_char``, ``c_int8_t``, ``c_int_least8_t`` or
``int8``) back into an object array. Previously such wrappers failed to
compile or to import.
