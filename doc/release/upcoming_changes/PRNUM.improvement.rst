F2PY reads more ``use`` statements and kinds
--------------------------------------------

* F2PY now parses a ``use`` statement that gives the module nature.
  One example is ``use, intrinsic :: iso_fortran_env``.
  The old parser dropped that statement.
  It printed "Could not crack the use statement".
* F2PY now combines every ``use`` of one module.
  The statements may be in one scope.
  They may also be in a module and in its procedures.
  The last statement used to replace the earlier ones.
  The names from the earlier statements were then lost.
* A ``complex`` entity whose kind is a real kind now uses the matching complex C type.
  This covers ``real32`` and ``real64`` from ``iso_fortran_env``, and ``c_float``, ``c_double`` and ``c_long_double`` from ``iso_c_binding``.
  Those entities used to be treated as single precision complex.
* A kind renamed on import now resolves to the intrinsic name.
  The forms are ``use iso_fortran_env, only: dp => real64`` and ``use iso_c_binding, only: dp => c_double``.
  ``real(dp)`` used to map to C ``float``.
  The Fortran wrappers of module procedures then failed to compile.
* F2PY now matches kind names in any case.
  ``REAL(KIND=REAL64)`` in a signature file maps to C ``double``.
  The signature file keeps the case of the source text.
