F2PY wrappers with ``intent(inout)`` one-byte integer scalars compile
---------------------------------------------------------------------

F2PY now defines the helper ``pyobj_from_signed_char1``.
Generated C code calls that helper for an ``intent(inout)`` scalar.
The scalar is a one-byte integer kind.
The kinds are ``integer*1``, ``c_signed_char``, ``c_int8_t``, ``c_int_least8_t`` and ``int8``.
The helper writes the scalar back into an object array.
Those wrappers used to fail while compiling.
Some builds imported a module whose link was missing the symbol.

``f2py -c`` honours more build options with the meson backend
-------------------------------------------------------------
* Extension modules are now linked by the Fortran compiler, which adds its
  runtime library. Modules built with LLVM ``flang`` no longer fail to import
  with undefined ``_Fortran*`` symbols.
* ``-D<macro>[=<value>]`` and ``-U<macro>`` are passed to the C and the Fortran
  compiler again, as they were with ``numpy.distutils``. This covers both the
  macros documented in the ``f2py`` usage text for the C wrapper, such as
  ``-DF2PY_REPORT_ON_ARRAY_COPY=<int>``, and macros used when preprocessing
  Fortran sources.
* ``--debug`` selects a debug build again. It used to be dropped, while any
  ``--f77flags`` or ``--f90flags`` value containing the word ``debug``
  selected a debug build instead.
* Relative ``-I`` and ``-L`` directories and extra object files in other
  directories are now found. They are resolved against the working directory,
  as the rest of the command line is, instead of the build directory.
* ``-I`` and ``--include-paths`` directories are searched in the order they are
  given. They were reordered at random, so an include file present in more
  than one of them could be taken from any of them.
