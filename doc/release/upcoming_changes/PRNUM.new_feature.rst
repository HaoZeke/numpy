``numpy-config`` and ``numpy.pc`` report the F2PY include directory
--------------------------------------------------------------------
``numpy-config --f2py-includedir`` prints the directory containing
``fortranobject.c`` and ``fortranobject.h``, the same path as
`numpy.f2py.get_include`. The ``numpy.pc`` pkg-config file provides it as
the ``f2py_includedir`` variable. In Meson, ``dependency('numpy')`` returns it
with ``get_variable(pkgconfig: 'f2py_includedir',
configtool: 'f2py-includedir')`` for either lookup method, so a
``meson.build`` no longer needs to run Python to find it.
