``f2py -c`` build options with the ``meson`` backend
----------------------------------------------------
* ``f2py -c`` no longer prints a message about the removed ``distutils``
  backend when ``--backend`` is not given.
* ``--debug`` builds the extension module with meson's ``debug`` build type
  again. Compiler flags passed with ``--f77flags`` or ``--f90flags`` no longer
  switch the build type to ``debug`` when they contain the word "debug".
* ``-D<macro>[=<value>]`` and ``-U<macro>`` are now passed on to the C
  compiler, so that macros such as ``F2PY_REPORT_ATEXIT`` and
  ``F2PY_REPORT_ON_ARRAY_COPY`` take effect.

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
