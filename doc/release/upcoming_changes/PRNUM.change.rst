``f2py`` wrappers no longer declare callback arguments twice
------------------------------------------------------------
The interface that ``f2py`` writes into the Fortran 90 wrapper of a routine
with a callback argument and an assumed-shape array declared the callback
``external`` and also gave it an interface body guessed from its call sites.
That specifies the ``EXTERNAL`` attribute twice, which Intel Fortran compilers
reject, and the guessed interface of a function callback could put an
``intent`` on its result, which ``gfortran`` rejects. The interface now
declares the callback only ``external``.
