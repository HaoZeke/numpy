``f2py`` test builds reject an already imported module name
-----------------------------------------------------------
``numpy.f2py.tests.util.build_module`` now raises ``RuntimeError`` when asked
to build an extension under a name that is already imported in the running
process, instead of returning the module that was imported first.
