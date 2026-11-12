``numpy.f2py.run_main`` leaves the settings of ``f2py`` as they were
--------------------------------------------------------------------
A run of ``f2py`` in a Python process, through ``numpy.f2py.run_main`` or ``numpy.f2py.main``, no longer changes the settings of the modules of ``f2py`` for the code that runs after it.
After a run without ``-h``, ``numpy.f2py.crackfortran`` kept the case of names.
After one with ``only:`` it read only those routines.
After one with ``--quiet`` it printed nothing, until the next run.
