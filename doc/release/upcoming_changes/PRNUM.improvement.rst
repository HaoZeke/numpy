``f2py -c`` no longer imports NumPy again to configure the Meson build
----------------------------------------------------------------------
The ``meson.build`` that ``f2py`` generates now contains the NumPy and F2PY
include directories of the NumPy running ``f2py``, instead of finding them by
importing ``numpy`` in a subprocess. A ``numpy`` directory next to the build
directory, such as a ``/tmp/numpy`` checkout when the default temporary build
directory is used, no longer breaks ``f2py -c``.
