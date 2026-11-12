``f2py`` runtime has C unit tests
---------------------------------
``fortranobject.c``, the C runtime compiled into every ``f2py`` extension module, now has unit tests written with `cmocka <https://cmocka.org>`__.
They are part of the ``f2py`` test suite and run when meson finds cmocka, for example from the ``libcmocka-dev`` (Debian, Ubuntu) or ``cmocka`` (conda-forge, Homebrew) packages.
Otherwise they are skipped.
