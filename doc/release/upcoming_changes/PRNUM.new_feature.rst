``f2py`` wraps allocatable array arguments
------------------------------------------
Fortran procedures with allocatable array dummy arguments, and functions
returning an allocatable array, can now be wrapped; previously the generated
wrappers did not compile. An ``intent(out)`` allocatable array is returned
with the shape allocated by the procedure, or as ``None`` if it is left
unallocated. ``intent(in)`` and ``intent(inout)`` arrays are copied into an
allocated array for the call, and ``intent(inout)`` arrays are written back.
