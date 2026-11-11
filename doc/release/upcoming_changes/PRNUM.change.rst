``f2py -c`` links extension modules with the Fortran compiler
-------------------------------------------------------------
The meson backend of ``f2py -c`` now links the extension module with the
Fortran compiler instead of the C compiler, so the Fortran compiler adds its
own runtime libraries and run paths. Modules built with LLVM Flang no longer
fail to import with undefined ``_Fortran*`` symbols, and the GCC runtime
libraries, such as ``libgomp``, are found on macOS without editing the run
paths of the module.
