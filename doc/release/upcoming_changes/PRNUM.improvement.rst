``f2py`` call-backs shared between routines
-------------------------------------------
Several routines of a module can now declare the same external ``intent(callback)``.
They share one call-back; previously the generated C code defined it once per routine and failed to compile.

A routine that calls such an external without declaring it now passes the Fortran arguments to the Python call-back, as the declaring routines do, instead of calling it with only ``<name>_extra_args``.
For backward compatibility, a Python function that can be called with the extra arguments alone, for instance because all of its parameters have default values, is still called with only those from such a routine.

Signature files written by ``f2py -h`` now keep the ``callback`` and ``hide`` intents of such call-backs, so modules built from them no longer fail to import or expose a hidden call-back as a required argument.
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
