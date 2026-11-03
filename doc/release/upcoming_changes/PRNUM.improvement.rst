``f2py`` buffer sizes use ``npy_intp``
--------------------------------------
Callback wrappers for an ``intent(inout)`` character argument now pass
``npy_intp`` dimensions to NumPy. They previously passed an ``int``
array, which compilers that reject incompatible pointer types, such as
GCC 14 and later, refused to build.

The lengths of character arguments and character function results are
now ``npy_intp`` in the generated wrappers and their helpers. Strings
longer than the largest C ``int`` no longer raise ``OverflowError``.
