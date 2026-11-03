``f2py`` no longer copies bool arrays for one-byte ``logical`` inputs
--------------------------------------------------------------------
NumPy bool arrays passed to ``intent(in)`` array arguments declared as
``logical(kind=1)``, ``logical*1`` or ``logical(c_bool)`` are now handed to the
Fortran routine directly instead of being copied to an ``int8`` array first.
Bool arrays are still converted for ``intent(in,out)`` and rejected for
``intent(inout)`` and ``intent(inplace)``, because Fortran may write to these
arguments and the Fortran standard leaves the representation of ``.TRUE.`` to
the compiler.
