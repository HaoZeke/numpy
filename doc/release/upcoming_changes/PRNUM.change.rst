``f2py`` runtime has C unit tests
---------------------------------
``fortranobject.c``, the C runtime compiled into every ``f2py`` extension module, now has unit tests written with `cmocka <https://cmocka.org>`__.
They are part of the ``f2py`` test suite and run when meson finds cmocka, for example from the ``libcmocka-dev`` (Debian, Ubuntu) or ``cmocka`` (conda-forge, Homebrew) packages.
Otherwise they are skipped.

``f2py`` reports command line errors and its backend accurately
---------------------------------------------------------------
``f2py`` now exits with status 2 when it is given an unknown option, the same
status ``argparse`` uses for the options it parses, and with status 1 when it
refuses to overwrite an existing signature file. Both used to exit with status
0, so scripts and build tools could not tell that nothing was generated.

``f2py -c`` no longer prints that it cannot use the distutils backend on every
run. ``--backend`` now defaults to ``meson``, and only ``--backend distutils``,
which still falls back to meson, says that the distutils backend was removed.
