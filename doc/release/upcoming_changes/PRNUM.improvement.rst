``f2py`` wrapper interfaces match the wrapped Fortran routine
-------------------------------------------------------------
A Fortran function without a ``result`` clause that takes an assumed-shape
argument now returns the correct value. The Fortran wrapper that ``f2py``
generates for it declared no type for the function, so the result was read
with an implicit type.

Building a routine with assumed-shape arguments from a signature file, such
as one written by ``f2py -h``, no longer fails to compile. The Fortran
wrapper no longer contains ``f2py``-only syntax such as ``check``,
``depend``, ``required``, ``intent(hide)``, argument default values,
``threadsafe`` or ``fortranname``, which also came from ``!f2py``
directives in Fortran sources. The wrapper of a function with a callback
argument no longer uses the ``__user__`` module that holds the callback
signature either.

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
