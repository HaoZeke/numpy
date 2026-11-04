``f2py`` explains tuples returned by call-back functions
--------------------------------------------------------
``f2py`` unpacks a tuple returned by a Python call-back function into
the call-back's return values.
Returning ``(a, b, c)`` from a call-back whose only return value is an
array therefore converted ``a`` alone and failed with an unrelated
dimension error.
If a call-back returns a tuple with more items than it has return values
and a return value then fails to convert, ``f2py`` now raises an error
that names the expected call-back signature and suggests returning a
list or an array.
The original conversion error is kept as its ``__cause__``.

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
