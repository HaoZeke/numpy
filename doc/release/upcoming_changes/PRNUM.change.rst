``f2py -c`` fixes for OpenMP, LLVM Flang and Intel ``ifx``
----------------------------------------------------------
The meson backend of ``f2py -c`` now links the extension module with the
Fortran compiler instead of the C compiler, so the Fortran compiler adds its
own runtime libraries and run paths. Modules built with LLVM Flang no longer
fail to import with undefined ``_Fortran*`` symbols, and the GCC runtime
libraries, such as ``libgomp``, are found on macOS without editing the run
paths of the module.

The extension modules also use default symbol visibility again, as they did
with ``numpy.distutils``. Under the hidden visibility that meson applies to
Python extension modules, Intel ``ifx`` hides calls to procedures with
explicit interfaces, such as the OpenMP routines in ``omp_lib``, and linking
failed with ``hidden symbol `omp_get_num_threads' isn't defined``.
