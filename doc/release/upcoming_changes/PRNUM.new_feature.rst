``f2py`` wraps allocatable array arguments
------------------------------------------

Fortran procedures with allocatable array dummy arguments can now be
wrapped.
Functions that return an allocatable array can now be wrapped too.
Previously the generated wrappers did not compile.
An ``intent(out)`` allocatable array is returned with the allocated
shape, or as ``None`` when left unallocated.
``intent(in)`` and ``intent(inout)`` arrays are copied into an allocated
array for the call, and ``intent(inout)`` arrays are written back.
