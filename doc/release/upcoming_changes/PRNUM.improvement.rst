``f2py -c`` can build in the directory that holds its inputs
------------------------------------------------------------
With the meson backend, ``f2py -c --build-dir DIR`` no longer fails with
``shutil.SameFileError`` when ``DIR`` already contains the inputs, such as
a ``.pyf`` signature file written there by ``f2py -h``, Fortran sources,
or object files, or when ``DIR`` is the current directory. Such inputs
are now used in place.
