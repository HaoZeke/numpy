``f2py -c`` resolves relative paths and links library files after objects
-------------------------------------------------------------------------
Directories given to ``f2py -c`` with ``-I`` and ``-L``, and object or
library files given by a relative path, are now resolved from the
directory in which ``f2py`` is run.
Previously the meson backend looked them up from its build directory, so
a command like ``f2py -c ext.pyf -Imods -Llib -lext`` failed because
meson could not find ``mods`` or ``lib``.

Library files (``.a``, ``.so`` and ``.dylib``) given to ``f2py -c`` are
now linked after the objects that use them.
Previously a static library given this way was linked too early to
provide any symbols, so the extension module built but failed to import
with an undefined symbol.
