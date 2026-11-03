``f2py -c`` resolves relative ``-I`` and ``-L`` paths from the current directory
--------------------------------------------------------------------------------
Directories given to ``f2py -c`` with ``-I`` and ``-L``, and object or library
files given by a relative path, are now resolved from the directory in which
``f2py`` is run. Previously the meson backend looked them up from its build
directory, so a command like ``f2py -c ext.pyf -Imods -Llib -lext`` failed
because meson could not find ``mods`` or ``lib``.
