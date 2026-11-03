``f2py -c`` passes ``-D`` and ``-U`` macros to the compilers again
------------------------------------------------------------------
The ``meson`` backend of ``f2py -c`` silently dropped preprocessor macros given
as ``-D<macro>[=<defn>]`` and ``-U<macro>``. They now reach both the C and the
Fortran compiler, as they did with the removed ``distutils`` backend. ``#ifdef``
blocks in preprocessed Fortran sources such as ``.F90`` files and the
documented C macros such as ``-DF2PY_REPORT_ON_ARRAY_COPY=<int>`` work again.
