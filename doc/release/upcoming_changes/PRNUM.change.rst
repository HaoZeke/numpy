``f2py`` wrappers no longer declare callback arguments twice
------------------------------------------------------------
The interface that ``f2py`` writes into the Fortran 90 wrapper of a routine
with a callback argument and an assumed-shape array declared the callback
``external`` and also gave it an interface body guessed from its call sites.
That specifies the ``EXTERNAL`` attribute twice, which Intel Fortran compilers
reject, and the guessed interface of a function callback could put an
``intent`` on its result, which ``gfortran`` rejects. The interface now
declares the callback only ``external``.

``f2py`` reports command line errors and its backend accurately
---------------------------------------------------------------
``f2py`` now exits with status 2 when it is given an unknown option, the same
status ``argparse`` uses for the options it parses, and with status 1 when it
refuses to overwrite an existing signature file. Both used to exit with status
0, so scripts and build tools could not tell that nothing was generated.

``f2py -c`` no longer prints that it cannot use the distutils backend on every
run. ``--backend`` now defaults to ``meson``, and only ``--backend distutils``,
which still falls back to meson, says that the distutils backend was removed.
