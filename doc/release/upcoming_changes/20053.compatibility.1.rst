``f2py`` no longer modifies read-only character arrays
------------------------------------------------------
``character`` and ``character(len=*)`` arguments with ``intent(inout)`` now raise ``ValueError`` when the array passed in is read-only, instead of overwriting its data.
