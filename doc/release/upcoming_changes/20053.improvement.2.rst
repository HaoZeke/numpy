``f2py`` no longer crashes on modules with nothing to wrap
----------------------------------------------------------
Importing an ``f2py``-built extension crashed when a Fortran module contained only derived types or interface blocks, and failed with ``SystemError`` when a COMMON block had only ``intent(hide)`` variables.
Such modules and COMMON blocks are now skipped, as modules without public entities already were.
