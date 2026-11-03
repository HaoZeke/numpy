``f2py`` wraps assumed-shape arguments that declare a lower bound
-----------------------------------------------------------------
Dummy arguments such as ``x(0:)`` or ``x(n:, 2:)`` used to produce C code that did not compile (``x_Dims[0]=1 + ;``), or a check that rejected every input when the lower bound was a variable.
They are now wrapped like ``x(:)``; the Fortran procedure still sees the lower bounds it declares.
