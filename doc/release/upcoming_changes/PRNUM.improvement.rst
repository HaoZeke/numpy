f2py routine objects pickle by reference
-----------------------------------------
``f2py`` routine objects now pickle by import path. ``pickle.loads``
returns the same object, including an F90 module routine such as
``m.t``. ``copy.copy`` and ``copy.deepcopy`` return that same object.
F90 module objects, module variable containers, and ``COMMON`` blocks
still raise ``TypeError``: they wrap Fortran memory. Copy the array
data out of those objects when a value must be pickled.

The extension must be regenerated and rebuilt with this F2PY support.
Installing a NumPy update does not replace the wrapper code embedded
in an existing extension binary. The defining extension must be
importable in the process which restores the routine.

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
