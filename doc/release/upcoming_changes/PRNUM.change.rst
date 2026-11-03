``f2py -c`` passes ``-D`` and ``-U`` macros to the compilers
------------------------------------------------------------
The ``meson`` backend of ``f2py -c`` used to drop ``-D<macro>[=<value>]`` and
``-U<macro>`` without a warning. They now reach both the C and the Fortran
compiler, so the documented ``-DPREPEND_FORTRAN``, ``-DNO_APPEND_FORTRAN``,
``-DUPPERCASE_FORTRAN``, ``-DF2PY_REPORT_ATEXIT`` and
``-DF2PY_REPORT_ON_ARRAY_COPY=<int>`` take effect again.
