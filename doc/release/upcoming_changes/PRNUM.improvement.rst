``f2py`` test builds reject an already imported module name
-----------------------------------------------------------
``numpy.f2py.tests.util.build_module`` now raises ``RuntimeError``.
The module name is already imported in the running process.
It used to return the module that was imported first.
