F2PY reads more ``use`` statements and kinds
--------------------------------------------

* F2PY now parses a ``use`` statement that gives the module nature.
  One example is ``use, intrinsic :: iso_fortran_env``.
  The old parser dropped that statement.
  It printed "Could not crack the use statement".
* F2PY now combines every ``use`` of one module.
  The statements may be in one scope.
  They may also be in a module and in its procedures.
  The last statement used to replace the earlier ones.
  The names from the earlier statements were then lost.
* A ``complex`` entity whose kind is a real kind now uses the matching complex C type.
  This covers ``real32`` and ``real64`` from ``iso_fortran_env``, and ``c_float``, ``c_double`` and ``c_long_double`` from ``iso_c_binding``.
  Those entities used to be treated as single precision complex.
* A kind renamed on import now resolves to the intrinsic name.
  The forms are ``use iso_fortran_env, only: dp => real64`` and ``use iso_c_binding, only: dp => c_double``.
  ``real(dp)`` used to map to C ``float``.
  The Fortran wrappers of module procedures then failed to compile.
* F2PY now matches kind names in any case.
  ``REAL(KIND=REAL64)`` in a signature file maps to C ``double``.
  The signature file keeps the case of the source text.

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
