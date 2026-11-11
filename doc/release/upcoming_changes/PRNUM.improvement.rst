``f2py -c`` links with Intel Fortran on Windows
-----------------------------------------------
With ``ifort`` or ``ifx`` on Windows, the ``meson`` backend now compiles Fortran sources with ``/names:lowercase`` and ``/assume:underscore``, the symbol names that the generated C wrappers call.
Linking no longer fails with unresolved external symbols such as ``fib_``.
To keep the Intel defaults, pass ``--f90flags="/names:uppercase /assume:nounderscore"`` together with ``-DUPPERCASE_FORTRAN -DNO_APPEND_FORTRAN``.
