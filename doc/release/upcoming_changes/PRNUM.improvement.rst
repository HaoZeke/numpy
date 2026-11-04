``f2py`` buffer sizes use ``npy_intp``
--------------------------------------
Callback wrappers for an ``intent(inout)`` character argument now pass
``npy_intp`` dimensions to NumPy. They previously passed an ``int``
array, which compilers that reject incompatible pointer types, such as
GCC 14 and later, refused to build.

The lengths of character arguments and character function results are
now ``npy_intp`` in the generated wrappers and their helpers. Strings
longer than the largest C ``int`` no longer raise ``OverflowError``.

Callback wrappers receive the hidden lengths of character arguments and
results as ``npy_intp``, the width that gfortran 8 and later and LLVM
flang pass. A callback that returns a character value now builds with
GCC 14 and later, and passes its result length rather than the address
of that length when the callback is a ``PyCapsule``.

Character array arguments of assumed length, such as
``character(len=*) :: a(n)``, now take their element size from a
``bytes`` or ``str`` input as ``npy_intp``. An element longer than the
largest C ``int`` previously raised ``SystemError``, or from 4 GiB on
was silently truncated; it is now passed whole.

Arrays with an initial value in a signature file, such as
``dimension(n) :: a = _i[0]``, are now filled using ``npy_intp``
indices, so ``_i[k]`` is an ``npy_intp``. Filling an array with more
elements than the largest C ``int`` previously overflowed the index and
wrote out of bounds.

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
