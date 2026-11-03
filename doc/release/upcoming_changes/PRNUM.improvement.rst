``f2py`` skips ``COMMON`` blocks whose members are all hidden
-------------------------------------------------------------
``COMMON`` blocks whose members are all ``intent(hide)`` are now skipped
instead of making the import fail with ``SystemError``.
