``f2py`` meson builds work with the Python environment in the project
---------------------------------------------------------------------
The ``meson.build`` that ``f2py -c`` writes, and keeps when given
``--build-dir``, now passes the NumPy and ``f2py`` include directories to Meson
as paths relative to the directory holding it. Meson rejects absolute paths
into its source tree, so configuring failed when the Python environment
providing NumPy was inside that tree, for example a ``.venv`` in a project
reusing the generated ``meson.build``.
