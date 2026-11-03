``f2py`` leaves out module variables it cannot wrap
---------------------------------------------------
A module variable of derived type is left out of the wrapper, with a message, instead of stopping ``f2py`` with ``KeyError: 'void'``.
A parameter whose size ``f2py`` cannot evaluate, such as one that is still implied-shape or still an assumed-length string, is left out the same way, instead of making the module fail to build or to import.
Named constants are copied when the module is initialized, so a value from LLVM ``flang`` is the constant rather than the temporary the compiler passed in.
