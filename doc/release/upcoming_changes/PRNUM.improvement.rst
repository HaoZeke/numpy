``f2py`` documents call-backs from separately built Fortran code
----------------------------------------------------------------
The ``f2py`` documentation now shows how Fortran code that is built
separately, for example into a static library, calls Python functions
through a wrapped entry point, both for call-backs passed as procedure
arguments and for call-backs with a fixed external name.

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
