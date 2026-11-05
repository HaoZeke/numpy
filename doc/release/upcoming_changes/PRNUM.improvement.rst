``f2py -c`` can build in the directory that holds its inputs
------------------------------------------------------------
``f2py -c --build-dir DIR`` no longer raises ``shutil.SameFileError``.
That used to happen when ``DIR`` already held an input.
A ``.pyf`` from ``f2py -h`` is one such input.
A Fortran source or an object file is another.
It also happened when ``DIR`` was the current directory.
Those inputs are now used in place.
