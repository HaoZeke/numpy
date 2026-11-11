F2PY wrapped Fortran code can be interrupted
--------------------------------------------
F2PY generated extension modules now provide the C function
``f2py_check_signals``, which long-running Fortran code can call through a
``bind(c)`` interface to run pending Python signal handlers. It returns a
nonzero value when a handler raised an exception, for example
`KeyboardInterrupt` after Ctrl-C; the Fortran code can then return early, and
the wrapper raises the exception. It can also be called from routines declared
``threadsafe``. See :ref:`f2py-interrupting`.
