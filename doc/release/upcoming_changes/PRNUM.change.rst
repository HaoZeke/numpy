``f2py`` compiler checks tolerate cleanup failures
--------------------------------------------------
The compiler checks in ``numpy.f2py.tests.util`` no longer raise when their
temporary ``meson`` directory cannot be removed, as happens on Windows while
another process still holds a file in it open. Previously the resulting
``PermissionError`` made every ``f2py`` test module fail to collect.
