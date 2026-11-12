``f2py`` gives each function ``ENTRY`` its own result type
----------------------------------------------------------
``f2py`` no longer raises ``KeyError`` for a Fortran function with an
``ENTRY`` statement whose name has no type declaration.
The entry is wrapped with the type given by the implicit typing rules.
In a function with a ``RESULT`` clause, an ``ENTRY`` is now wrapped with
its own type instead of the type of the ``RESULT`` variable.
