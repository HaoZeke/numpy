``f2py -c`` fixes for OpenMP, LLVM Flang and Intel ``ifx``
----------------------------------------------------------

The meson backend of ``f2py -c`` now links the extension with the Fortran compiler.
The Fortran compiler adds its own runtime libraries and run paths.
A module built with LLVM Flang now imports.
The undefined ``_Fortran*`` symbols are gone.
On macOS the GCC runtime libraries, including ``libgomp``, are found without an edit to the module's run paths.

The extension modules use default symbol visibility again.
``numpy.distutils`` used that visibility.
Meson otherwise gives Python extension modules hidden visibility.
Intel ``ifx`` then hides a call to a procedure that has an explicit interface.
The OpenMP routines in ``omp_lib`` are such procedures.
The link failed with ``hidden symbol `omp_get_num_threads' isn't defined``.

``--debug`` again selects the meson ``debug`` build type.
The ``numpy.distutils`` options ``--opt``, ``--arch``, ``--noopt``, ``--noarch``, ``--f77exec``, ``--f90exec`` and ``--help-fcompiler`` do nothing under meson.
``f2py -c`` now prints that each of those options is ignored.

The meson build of ``f2py -c`` also compiles ``fortranobject.c`` as a static library with hidden visibility.
The extension links that library with ``link_with``.
The helpers in that file are not exported.
The extension module itself keeps default visibility.

``f2py`` reports command line errors and its backend accurately
---------------------------------------------------------------
``f2py`` now exits with status 2 when it is given an unknown option, the same
status ``argparse`` uses for the options it parses, and with status 1 when it
refuses to overwrite an existing signature file. Both used to exit with status
0, so scripts and build tools could not tell that nothing was generated.

``f2py -c`` no longer prints that it cannot use the distutils backend on every
run. ``--backend`` now defaults to ``meson``, and only ``--backend distutils``,
which still falls back to meson, says that the distutils backend was removed.
