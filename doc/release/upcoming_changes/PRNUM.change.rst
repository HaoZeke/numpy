``f2py`` tests no longer index empty Fortran arrays
---------------------------------------------------
The Fortran helpers of ``TestDimSpec`` in ``numpy/f2py/tests/test_crackfortran.py`` no longer reference an element of a zero-size array.
The out-of-bounds reference was reported by valgrind as a use of uninitialised memory, and it aborted the test run when the tests were compiled with ``gfortran -fcheck=bounds``.
