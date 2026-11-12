LFortran builds do not pass ``-fvisibility``
--------------------------------------------
``gnu_symbol_visibility`` makes Meson add ``-fvisibility`` to the
Fortran compile line. LFortran rejects that flag. The keyword stays
for the other compilers, which need it so ifx can bind shared
libraries, and is omitted when the compiler is LFortran.
