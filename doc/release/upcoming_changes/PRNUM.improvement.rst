``f2py`` converts character array results of call-backs to the declared length
-------------------------------------------------------------------------------
A character array returned by a Python call-back for an ``intent(out)``
argument is now converted to the length declared in the call-back signature,
as for the array arguments of the wrapped routine itself. Previously the
strings kept the length the call-back produced and were copied into the
Fortran array byte for byte, so shorter strings ran into the neighbouring
elements.

``f2py`` call-backs that raise no longer abort with Python thread-local storage
-------------------------------------------------------------------------------
When the call-back pointers are kept in the Python thread state (with
``F2PY_USE_PYTHON_TLS`` defined, or with a C compiler without thread-local
storage), an exception raised by a call-back is now propagated to the caller
of the wrapped routine. Previously it aborted the interpreter with
``Fatal Python error: F2PySwapThreadLocalCallbackPtr: PyLong_AsVoidPtr failed``.

``f2py`` modules with character function call-backs build with current compilers
---------------------------------------------------------------------------------
The code generated for a call-back that is a character function passed the
address of the result length where the length itself is expected, which GCC 14
and Clang 15 and newer reject, so such modules failed to build. The length is
now passed by value, which also gives a ``PyCapsule`` passed as the call-back
the correct length.

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
