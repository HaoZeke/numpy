``f2py`` rejects the options of the removed ``distutils`` backend
-----------------------------------------------------------------
``f2py`` now exits with an error that names the replacement when it is passed
``--backend distutils`` or one of the build options that only the removed
``distutils`` backend understood: ``--fcompiler``, ``--compiler``,
``--f77exec``, ``--f90exec``, ``--help-fcompiler``, ``--opt``, ``--arch``,
``--noopt``, ``--noarch``, ``--link-<resource>`` and ``--help-link``.
``f2py -c`` used to ignore them, so that for example ``--f90exec`` did not
change the Fortran compiler used for the build. Select the compilers with the
``FC`` and ``CC`` environment variables, pass compiler flags with
``--f77flags`` or ``--f90flags``, and link dependencies with ``--dep``.

External array-valued functions use explicit interfaces
------------------------------------------------------------
F2PY generates an explicit Fortran interface when wrapping an external
function with an array result, including functions with fixed-shape
arguments. This preserves the result's array rank at the generated call
site. In particular, a CHARACTER array result is not declared as a scalar
CHARACTER result whose length is the array extent.
