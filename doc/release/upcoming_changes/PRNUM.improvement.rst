``f2py`` tests check for compilers only when a test needs one
-------------------------------------------------------------
Importing the ``numpy.f2py`` test modules no longer runs ``meson`` to look for
C and Fortran compilers. The check now runs in a session-scoped ``pytest``
fixture the first time a test builds an extension module, so collecting the
tests is faster and a failing check can no longer abort collection. Tests
which do not compile anything, such as the parser and command line tests, now
run when ``meson`` is not installed instead of being skipped.
