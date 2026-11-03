``f2py`` folds element ``DATA`` into the named array
----------------------------------------------------
``f2py`` used to read ``data d(1) / 1 /`` as a new variable named ``d(1)``.
A one-dimensional array with a constant extent now receives
``(/v1, v2, .../)`` once every element is known.
Other element ``DATA`` forms are left for the Fortran compiler to apply.
