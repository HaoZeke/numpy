``f2py`` resolves ``kind`` parameters renamed on ``use``
--------------------------------------------------------
``f2py`` now follows renames such as ``use iso_fortran_env, only: dp => real64``
when it maps a ``kind`` to a C type, so ``complex(kind=dp)`` wraps as double
precision complex instead of single precision complex. Renamed ``iso_c_binding``
kinds and renamed parameters of other wrapped modules resolve the same way.
``complex(kind=real64)`` from ``iso_fortran_env`` and ``complex(kind=c_double)``
from ``iso_c_binding`` also map to double precision complex now. Previously,
passing a ``complex128`` array to such an argument was rejected or overflowed the
buffer handed to Fortran. ``use`` statements that name the module nature, such as
``use, intrinsic :: iso_fortran_env``, are no longer ignored.
