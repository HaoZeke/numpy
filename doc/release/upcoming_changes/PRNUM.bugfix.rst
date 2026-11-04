Fortran module parameters retain their values with LLVM Flang
------------------------------------------------------------
F2PY copies module PARAMETER values during initialization into aligned
static storage rather than retaining compiler-owned temporary addresses.
Parameter arrays are read-only, while mutable module variables retain
bidirectional access to their Fortran storage.
