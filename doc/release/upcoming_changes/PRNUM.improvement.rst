``f2py -c`` build options with the ``meson`` backend
----------------------------------------------------
* ``f2py -c`` no longer prints a message about the removed ``distutils``
  backend when ``--backend`` is not given.
* ``--debug`` builds the extension module with meson's ``debug`` build type
  again. Compiler flags passed with ``--f77flags`` or ``--f90flags`` no longer
  switch the build type to ``debug`` when they contain the word "debug".
* ``-D<macro>[=<value>]`` and ``-U<macro>`` are now passed on to the C
  compiler, so that macros such as ``F2PY_REPORT_ATEXIT`` and
  ``F2PY_REPORT_ON_ARRAY_COPY`` take effect.
