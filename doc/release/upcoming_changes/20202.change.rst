``f2py`` handles the Fortran 2008 ``BLOCK`` construct
-----------------------------------------------------
``END BLOCK`` no longer ends the procedure that contains a ``BLOCK``
construct, so the procedures that follow it in the same module or file are
wrapped instead of being dropped or breaking the build.
