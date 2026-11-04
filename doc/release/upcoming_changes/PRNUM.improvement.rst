``f2py`` no longer crashes on Fortran objects with nothing to wrap
------------------------------------------------------------------
Reading ``_cpointer`` from an ``f2py`` object without data, such as an
unallocated allocatable array that is the only variable of a Fortran 90
module, or a routine declared with an empty ``fortranname``, now raises
``AttributeError`` instead of crashing.
Fortran 90 modules that only define derived types or interfaces are now
skipped, like modules without public entities, instead of crashing on
import.
``COMMON`` blocks whose members are all ``intent(hide)`` are now skipped
instead of making the import fail with ``SystemError``.

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
