``f2py -c`` modules can call shared-library procedures with Intel Fortran
-------------------------------------------------------------------------
Extension modules built with ``f2py -c`` are compiled with default symbol visibility again, as they were with the removed ``distutils`` backend.
Meson hides the symbols of Python extension modules by default, and Intel's ``ifx`` applies ``-fvisibility=hidden`` to external references as well as definitions, so Fortran code that called procedures from shared libraries through explicit interfaces, such as the OpenMP routines in ``omp_lib``, failed to link with ``hidden symbol ... isn't defined``.
