Deleting ``f2py`` module data raises ``AttributeError``
-------------------------------------------------------
``del`` on a variable of an ``f2py`` wrapped Fortran module or common block, for example ``del mod.x``, crashed the interpreter.
It now raises ``AttributeError``.
To deallocate an allocatable array, assign ``None`` to it.
