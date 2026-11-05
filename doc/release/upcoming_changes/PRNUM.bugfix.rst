Keep f2py module constants in persistent native storage
----------------------------------------------------
Module ``PARAMETER`` values exposed by f2py retain compiler-typed
storage after module initialization, including constants materialized
as temporary arguments by LLVM Flang. Mutable module variables retain
their shared Fortran storage.
