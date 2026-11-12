``f2py`` extension modules report errors raised during import
-------------------------------------------------------------
The initialization code of ``f2py``-built extension modules no longer ignores failures such as a ``MemoryError``, which could crash the interpreter or turn into a ``SystemError`` that hid the original exception.
``F2PyDict_SetItemString`` and ``F2PyCapsule_FromVoidPtr`` from ``fortranobject.h`` no longer print or clear the pending exception.
