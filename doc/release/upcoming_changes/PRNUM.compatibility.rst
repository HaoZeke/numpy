``f2py -c`` passes ``-D`` and ``-U`` options to the compilers again
-------------------------------------------------------------------
The ``meson`` backend accepted ``-D<macro>[=<defn>]`` and ``-U<macro>`` but
silently dropped them, so options such as ``-DF2PY_REPORT_ON_ARRAY_COPY=1`` or
``-DNO_APPEND_FORTRAN`` had no effect. They are now passed to the C and Fortran
compilers, as ``numpy.distutils`` used to do. Build commands that still carry
such options from older scripts may need to drop the ones that do not apply to
the compiler in use.

External array-valued functions use explicit interfaces
------------------------------------------------------------
F2PY generates an explicit Fortran interface when wrapping an external
function with an array result, including functions with fixed-shape
arguments. This preserves the result's array rank at the generated call
site. In particular, a CHARACTER array result is not declared as a scalar
CHARACTER result whose length is the array extent.
